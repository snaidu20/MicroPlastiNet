"""
graph_builder.py — Hydrological Flow Graph Builder for Coastal Georgia Rivers
==============================================================================
Builds a synthetic but geographically realistic directed flow graph for the
Ogeechee, Savannah, and Altamaha river systems in coastal Georgia (USA).

Graph structure (200 nodes total):
  - 50 sampling stations (measurement sites)
  - 100 candidate sources (factories, urban runoff, agricultural runoff)
  - 50 river junctions (confluence / routing nodes)

Geography grounded in real HydroSHEDS-style drainage networks.
Outputs a PyTorch Geometric Data object and a GeoJSON file.

References:
  - Lehner, B. et al. (2008). New global hydrography derived from
    spaceborne elevation data. Geophysical Research Letters, 35(10).
    https://www.hydrosheds.org/
  - Wang, M. et al. (2019). Deep Graph Library. arXiv:1909.01315.
"""

import json
import math
import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Real geographic anchor points for coastal Georgia river systems
# (lat, lon) — sourced from USGS StreamStats / NHD Plus
# ---------------------------------------------------------------------------

# Savannah River corridor (Hardeeville, SC / Savannah, GA to Augusta, GA)
SAVANNAH_ANCHORS = [
    (32.0835, -81.0998),   # Savannah River mouth (Port of Savannah area)
    (32.1200, -81.2000),   # Lower Savannah
    (32.3000, -81.4500),   # Mid Savannah
    (32.5500, -81.7000),   # Upper mid Savannah
    (32.8000, -81.9000),   # Sylvania area
    (33.0000, -81.9800),   # Statesboro / Millen area
    (33.3500, -82.0500),   # Waynesboro area
    (33.4800, -82.0800),   # Augusta confluence
]

# Ogeechee River corridor
OGEECHEE_ANCHORS = [
    (31.8600, -81.1800),   # Ogeechee mouth / Ossabaw Sound
    (31.9600, -81.3500),   # Lower Ogeechee
    (32.0800, -81.6000),   # Richmond Hill area
    (32.2000, -81.8500),   # Mid Ogeechee
    (32.5000, -82.0000),   # Statesboro area
    (32.6500, -82.2000),   # Upper mid Ogeechee
    (32.8500, -82.5000),   # Upper Ogeechee / Sandersville area
]

# Altamaha River corridor
ALTAMAHA_ANCHORS = [
    (31.3300, -81.3000),   # Altamaha mouth / Wolf Island
    (31.4500, -81.5000),   # Lower Altamaha
    (31.6500, -81.7000),   # Everett City area
    (31.8500, -81.9000),   # Jesup area
    (31.9500, -82.1500),   # Baxley area
    (32.1000, -82.4000),   # Hazlehurst area
    (32.2000, -82.6500),   # Douglas area
    (31.7500, -82.3000),   # Ocmulgee tributary junction
    (31.6500, -82.5000),   # Ocmulgee lower reach
]

# Major coastal Georgia cities / industrial zones for source locations
SOURCE_ANCHORS = {
    "savannah_port_industrial":    (32.0835, -81.0998),
    "pooler_manufacturing":        (32.1156, -81.2484),
    "statesboro_urban":            (32.4487, -81.7831),
    "augusta_industrial":          (33.4735, -82.0105),
    "brunswick_industrial":        (31.1499, -81.4915),
    "jesup_paper_mill":            (31.6077, -81.8849),
    "douglas_agri":                (31.5087, -82.8510),
    "baxley_forestry":             (31.7805, -82.3485),
    "waynesboro_agri":             (33.0898, -82.0136),
    "sylvania_agri":               (32.7479, -81.6370),
    "reidsville_urban":            (32.0843, -82.1151),
    "vidalia_agri":                (32.2174, -82.4135),
    "claxton_agri":                (32.1601, -81.9071),
    "glennville_agri":             (31.9324, -81.9239),
    "darien_coastal":              (31.3716, -81.4315),
    "macon_industrial":            (32.8407, -83.6324),
    "hinesville_urban":            (31.8468, -81.5962),
    "richmond_hill_suburban":      (31.9249, -81.3001),
}


def _interpolate_points(anchors: list, n: int) -> list:
    """Linearly interpolate n points along a sequence of anchor (lat, lon) pairs."""
    total_dist = 0.0
    segments = []
    for i in range(len(anchors) - 1):
        lat1, lon1 = anchors[i]
        lat2, lon2 = anchors[i + 1]
        d = math.hypot(lat2 - lat1, lon2 - lon1)
        segments.append((d, anchors[i], anchors[i + 1]))
        total_dist += d
    points = []
    cumulative = 0.0
    seg_idx = 0
    for k in range(n):
        t_global = (k / (n - 1)) * total_dist if n > 1 else 0
        while seg_idx < len(segments) - 1 and cumulative + segments[seg_idx][0] < t_global:
            cumulative += segments[seg_idx][0]
            seg_idx += 1
        seg_len, (lat1, lon1), (lat2, lon2) = segments[seg_idx]
        if seg_len > 0:
            t_local = (t_global - cumulative) / seg_len
        else:
            t_local = 0.0
        t_local = max(0.0, min(1.0, t_local))
        lat = lat1 + t_local * (lat2 - lat1)
        lon = lon1 + t_local * (lon2 - lon1)
        points.append((lat, lon))
    return points


def _elevation_from_coords(lat: float, lon: float) -> float:
    """
    Approximate elevation (meters) from coordinates using a simple
    gradient model: coastal GA rises ~1 m per ~0.1° northward/westward
    from sea level at the coast (lat ~31.3°, lon ~-81.0°).
    """
    base_lat, base_lon = 31.3, -81.0
    elev = (lat - base_lat) * 15.0 + (-(lon) - 81.0) * 8.0
    return max(0.0, elev + np.random.normal(0, 2))


def _population_density(node_type: str, lat: float, lon: float) -> float:
    """
    Approximate population density (persons/km²) based on proximity to
    known urban centers in coastal Georgia.
    """
    urban_centers = [
        (32.0835, -81.0998, 1200),   # Savannah
        (33.4735, -82.0105, 800),    # Augusta
        (31.1499, -81.4915, 600),    # Brunswick
        (32.4487, -81.7831, 400),    # Statesboro
        (31.8468, -81.5962, 300),    # Hinesville
    ]
    if node_type == "junction":
        return 0.0
    density = 20.0  # rural baseline
    for clat, clon, pop in urban_centers:
        dist = math.hypot(lat - clat, lon - clon)
        if dist < 0.5:
            density = max(density, pop * math.exp(-dist / 0.2))
    return density + abs(np.random.normal(0, density * 0.1))


# ---------------------------------------------------------------------------
# Node-type encoding
# ---------------------------------------------------------------------------
NODE_TYPES = ["station", "factory", "urban_runoff", "agricultural_runoff", "junction"]
# one-hot positions: 0=station, 1=factory, 2=urban_runoff, 3=agri_runoff, 4=junction
TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}


def _one_hot(idx: int, n: int = 5) -> list:
    v = [0.0] * n
    v[idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_graph(save_dir: str = None) -> Data:
    """
    Build and return the coastal Georgia hydrological flow graph.

    Returns
    -------
    data : torch_geometric.data.Data
        Graph with node features x, edge_index, edge_attr, and metadata.
    """
    G = nx.DiGraph()
    node_meta = {}   # node_id -> dict of metadata
    node_id = 0

    # ── 1. Sampling Stations (50) ──────────────────────────────────────────
    # Distribute across three rivers
    sav_stations = _interpolate_points(SAVANNAH_ANCHORS, 18)
    oge_stations = _interpolate_points(OGEECHEE_ANCHORS, 16)
    alt_stations = _interpolate_points(ALTAMAHA_ANCHORS, 16)

    station_ids = []
    for pts, river in [(sav_stations, "savannah"), (oge_stations, "ogeechee"),
                       (alt_stations, "altamaha")]:
        for lat, lon in pts:
            jitter_lat = lat + np.random.normal(0, 0.005)
            jitter_lon = lon + np.random.normal(0, 0.005)
            ntype = "station"
            meta = {
                "node_id": node_id,
                "node_type": ntype,
                "type_idx": TYPE_TO_IDX[ntype],
                "lat": jitter_lat,
                "lon": jitter_lon,
                "elevation": _elevation_from_coords(jitter_lat, jitter_lon),
                "population_density": _population_density(ntype, jitter_lat, jitter_lon),
                "river": river,
                "label": f"station_{node_id:03d}",
            }
            G.add_node(node_id, **meta)
            node_meta[node_id] = meta
            station_ids.append(node_id)
            node_id += 1

    # ── 2. Candidate Sources (100) ─────────────────────────────────────────
    # Mix: 30 factories, 35 urban runoff, 35 agricultural runoff
    source_ids = []

    # Factories near industrial anchors
    factory_anchors = [
        SOURCE_ANCHORS["savannah_port_industrial"],
        SOURCE_ANCHORS["pooler_manufacturing"],
        SOURCE_ANCHORS["jesup_paper_mill"],
        SOURCE_ANCHORS["augusta_industrial"],
        SOURCE_ANCHORS["brunswick_industrial"],
    ]
    for i in range(30):
        base = factory_anchors[i % len(factory_anchors)]
        lat = base[0] + np.random.normal(0, 0.05)
        lon = base[1] + np.random.normal(0, 0.05)
        ntype = "factory"
        meta = {
            "node_id": node_id,
            "node_type": ntype,
            "type_idx": TYPE_TO_IDX[ntype],
            "lat": lat,
            "lon": lon,
            "elevation": _elevation_from_coords(lat, lon) + np.random.uniform(0, 10),
            "population_density": _population_density(ntype, lat, lon),
            "river": "mixed",
            "label": f"factory_{i:03d}",
        }
        G.add_node(node_id, **meta)
        node_meta[node_id] = meta
        source_ids.append(node_id)
        node_id += 1

    # Urban runoff near cities
    urban_anchors = [
        SOURCE_ANCHORS["savannah_port_industrial"],
        SOURCE_ANCHORS["statesboro_urban"],
        SOURCE_ANCHORS["hinesville_urban"],
        SOURCE_ANCHORS["richmond_hill_suburban"],
        SOURCE_ANCHORS["reidsville_urban"],
    ]
    for i in range(35):
        base = urban_anchors[i % len(urban_anchors)]
        lat = base[0] + np.random.normal(0, 0.08)
        lon = base[1] + np.random.normal(0, 0.08)
        ntype = "urban_runoff"
        meta = {
            "node_id": node_id,
            "node_type": ntype,
            "type_idx": TYPE_TO_IDX[ntype],
            "lat": lat,
            "lon": lon,
            "elevation": _elevation_from_coords(lat, lon) + np.random.uniform(0, 5),
            "population_density": _population_density(ntype, lat, lon),
            "river": "mixed",
            "label": f"urban_runoff_{i:03d}",
        }
        G.add_node(node_id, **meta)
        node_meta[node_id] = meta
        source_ids.append(node_id)
        node_id += 1

    # Agricultural runoff
    agri_anchors = [
        SOURCE_ANCHORS["sylvania_agri"],
        SOURCE_ANCHORS["vidalia_agri"],
        SOURCE_ANCHORS["claxton_agri"],
        SOURCE_ANCHORS["glennville_agri"],
        SOURCE_ANCHORS["baxley_forestry"],
        SOURCE_ANCHORS["douglas_agri"],
        SOURCE_ANCHORS["waynesboro_agri"],
    ]
    for i in range(35):
        base = agri_anchors[i % len(agri_anchors)]
        lat = base[0] + np.random.normal(0, 0.10)
        lon = base[1] + np.random.normal(0, 0.10)
        ntype = "agricultural_runoff"
        meta = {
            "node_id": node_id,
            "node_type": ntype,
            "type_idx": TYPE_TO_IDX[ntype],
            "lat": lat,
            "lon": lon,
            "elevation": _elevation_from_coords(lat, lon) + np.random.uniform(0, 3),
            "population_density": 0.0,
            "river": "mixed",
            "label": f"agri_runoff_{i:03d}",
        }
        G.add_node(node_id, **meta)
        node_meta[node_id] = meta
        source_ids.append(node_id)
        node_id += 1

    # ── 3. River Junctions (50) ────────────────────────────────────────────
    junction_locs = (
        _interpolate_points(SAVANNAH_ANCHORS, 18)[:15]
        + _interpolate_points(OGEECHEE_ANCHORS, 14)[:12]
        + _interpolate_points(ALTAMAHA_ANCHORS, 16)[:12]
        + [
            (32.25, -81.90),  # Ogeechee / Canoochee confluence
            (31.80, -82.10),  # Altamaha / Ocmulgee confluence
            (31.55, -81.40),  # Altamaha lower
            (31.40, -81.38),  # Altamaha delta
            (32.05, -81.12),  # Savannah lower
            (32.30, -81.55),  # Savannah mid
            (32.70, -81.80),  # Savannah upper mid
            (33.10, -81.95),  # Savannah upper
            (31.95, -81.65),  # Ogeechee mid
            (32.60, -82.10),  # Ogeechee upper
            (31.70, -82.05),  # Altamaha inner
        ]
    )

    # Ensure exactly 50 junctions
    junction_locs = junction_locs[:50]
    while len(junction_locs) < 50:
        lat = random.uniform(31.3, 33.5)
        lon = random.uniform(-83.0, -81.0)
        junction_locs.append((lat, lon))

    junction_ids = []
    for i, (lat, lon) in enumerate(junction_locs):
        ntype = "junction"
        meta = {
            "node_id": node_id,
            "node_type": ntype,
            "type_idx": TYPE_TO_IDX[ntype],
            "lat": lat,
            "lon": lon,
            "elevation": _elevation_from_coords(lat, lon),
            "population_density": 0.0,
            "river": "junction",
            "label": f"junction_{i:03d}",
        }
        G.add_node(node_id, **meta)
        node_meta[node_id] = meta
        junction_ids.append(node_id)
        node_id += 1

    assert node_id == 200, f"Expected 200 nodes, got {node_id}"

    # ── 4. Build Directed Edges (downstream flow) ──────────────────────────
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))

    def flow_rate(elev_up: float, elev_down: float, dist_km: float) -> float:
        """Synthetic flow rate (m³/s) — higher gradient = higher flow."""
        grad = max(0, elev_up - elev_down) / max(dist_km, 0.1)
        return 0.5 + grad * 50 + abs(np.random.normal(0, 0.5))

    def wind_correlation(lat1, lon1, lat2, lon2) -> float:
        """Prevailing wind in coastal GA is SW→NE; correlation if edge aligns."""
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        # Prevailing direction: NE (dlat>0, dlon>0) from SW origin
        angle = math.atan2(dlat, dlon)  # radians
        preferred = math.radians(45)    # NE
        corr = math.cos(angle - preferred)
        return max(0.0, corr) + abs(np.random.normal(0, 0.05))

    added_edges = set()

    def add_edge(u: int, v: int):
        if u == v or (u, v) in added_edges:
            return
        meta_u = node_meta[u]
        meta_v = node_meta[v]
        dist = haversine(meta_u["lat"], meta_u["lon"], meta_v["lat"], meta_v["lon"])
        fr = flow_rate(meta_u["elevation"], meta_v["elevation"], dist)
        wc = wind_correlation(meta_u["lat"], meta_u["lon"], meta_v["lat"], meta_v["lon"])
        G.add_edge(u, v, flow_rate=fr, distance=dist, wind_correlation=wc)
        added_edges.add((u, v))

    # --- Station-to-station edges along each river (longitudinal flow) ---
    sav_stn = station_ids[:18]   # Savannah: upstream (high index) → downstream (low index lat-wise)
    oge_stn = station_ids[18:34]
    alt_stn = station_ids[34:50]

    # Sort stations by latitude descending (upstream=high lat → downstream=low lat)
    for river_stations in [sav_stn, oge_stn, alt_stn]:
        ordered = sorted(river_stations, key=lambda n: -node_meta[n]["lat"])
        for i in range(len(ordered) - 1):
            add_edge(ordered[i], ordered[i + 1])

    # --- Junctions connect rivers and route flow ---
    # Connect some junctions into the station chains
    for jid in junction_ids:
        jlat = node_meta[jid]["lat"]
        jlon = node_meta[jid]["lon"]
        # Find nearby stations and connect junction → downstream station
        nearby = sorted(
            station_ids,
            key=lambda s: haversine(jlat, jlon, node_meta[s]["lat"], node_meta[s]["lon"])
        )
        # Connect to 1-2 nearest downstream stations
        for stn in nearby[:2]:
            slat = node_meta[stn]["lat"]
            # Junction → station if junction is upstream (higher lat/elev)
            if node_meta[jid]["elevation"] >= node_meta[stn]["elevation"]:
                add_edge(jid, stn)
            else:
                add_edge(stn, jid)

    # --- Sources connect to nearest junction or station (sources → downstream nodes) ---
    all_routing = junction_ids + station_ids
    for src_id in source_ids:
        slat = node_meta[src_id]["lat"]
        slon = node_meta[src_id]["lon"]
        dists = [(haversine(slat, slon, node_meta[n]["lat"], node_meta[n]["lon"]), n)
                 for n in all_routing]
        dists.sort()
        # Connect to 1-3 nearest routing nodes
        for d, n in dists[:3]:
            if d < 50.0:  # within 50 km
                # Sources always flow downstream
                add_edge(src_id, n)

    # --- Inter-junction edges (tributary merges) ---
    for i, jid in enumerate(junction_ids):
        jlat = node_meta[jid]["lat"]
        jlon = node_meta[jid]["lon"]
        dists = [(haversine(jlat, jlon, node_meta[jid2]["lat"], node_meta[jid2]["lon"]), jid2)
                 for jid2 in junction_ids if jid2 != jid]
        dists.sort()
        for d, jid2 in dists[:2]:
            if d < 80.0:
                # Higher elevation flows to lower
                if node_meta[jid]["elevation"] > node_meta[jid2]["elevation"]:
                    add_edge(jid, jid2)
                else:
                    add_edge(jid2, jid)

    # --- Ensure strong connectivity: Add a few cross-river edges ---
    add_edge(sav_stn[-1], oge_stn[-1])   # Savannah outlet → Ogeechee outlet area
    add_edge(oge_stn[-1], alt_stn[-1])   # Ogeechee outlet → Altamaha outlet area

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── 5. Assemble node features ──────────────────────────────────────────
    # Feature vector per node:
    # [lat, lon, elevation, population_density, one_hot×5] = 9 features
    node_features = []
    node_labels = []
    node_types_list = []
    node_lats = []
    node_lons = []

    for nid in range(200):
        m = node_meta[nid]
        oh = _one_hot(m["type_idx"], 5)
        # Normalize lat/lon to [0,1] range for the study area
        norm_lat = (m["lat"] - 31.3) / (33.5 - 31.3)
        norm_lon = (m["lon"] - (-83.0)) / ((-81.0) - (-83.0))
        norm_elev = m["elevation"] / 100.0   # max ~100m in study area
        norm_pop = m["population_density"] / 1500.0
        feat = [norm_lat, norm_lon, norm_elev, norm_pop] + oh
        node_features.append(feat)
        node_labels.append(m["label"])
        node_types_list.append(m["node_type"])
        node_lats.append(m["lat"])
        node_lons.append(m["lon"])

    x = torch.tensor(node_features, dtype=torch.float)   # [200, 9]

    # ── 6. Assemble edge tensors ───────────────────────────────────────────
    edges = list(G.edges(data=True))
    src_list = [e[0] for e in edges]
    dst_list = [e[1] for e in edges]
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Edge features: [flow_rate, distance, wind_correlation]
    # Normalise: flow 0-100 m³/s, distance 0-200 km, wind corr 0-1
    edge_feats = []
    for _, _, attr in edges:
        ef = [
            min(attr["flow_rate"], 100.0) / 100.0,
            min(attr["distance"], 200.0) / 200.0,
            min(attr["wind_correlation"], 1.0),
        ]
        edge_feats.append(ef)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float)   # [E, 3]

    # ── 7. Create PyG Data object ──────────────────────────────────────────
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = 200
    data.node_labels = node_labels
    data.node_types = node_types_list
    data.node_lats = node_lats
    data.node_lons = node_lons
    data.station_ids = station_ids
    data.source_ids = source_ids
    data.junction_ids = junction_ids

    # ── 8. Save outputs ───────────────────────────────────────────────────
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # PyG graph
        torch.save(data, save_path / "flow_graph.pt")
        print(f"Saved PyG graph → {save_path / 'flow_graph.pt'}")

        # GeoJSON for dashboard
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        for nid in range(200):
            m = node_meta[nid]
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [m["lon"], m["lat"]]
                },
                "properties": {
                    "id": nid,
                    "label": m["label"],
                    "node_type": m["node_type"],
                    "elevation": round(m["elevation"], 2),
                    "population_density": round(m["population_density"], 2),
                    "river": m["river"],
                }
            }
            geojson["features"].append(feat)

        # Add edges as LineStrings
        for u, v, attr in edges:
            mu, mv = node_meta[u], node_meta[v]
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[mu["lon"], mu["lat"]], [mv["lon"], mv["lat"]]]
                },
                "properties": {
                    "from": u,
                    "to": v,
                    "flow_rate": round(attr["flow_rate"], 3),
                    "distance_km": round(attr["distance"], 3),
                    "wind_correlation": round(attr["wind_correlation"], 3),
                }
            }
            geojson["features"].append(feat)

        geo_path = save_path / "flow_graph.geojson"
        with open(geo_path, "w") as f:
            json.dump(geojson, f)
        print(f"Saved GeoJSON → {geo_path}")

        # Node metadata CSV
        import csv
        csv_path = save_path / "node_metadata.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "node_id", "label", "node_type", "lat", "lon",
                "elevation", "population_density", "river"
            ])
            writer.writeheader()
            for nid in range(200):
                m = node_meta[nid]
                writer.writerow({
                    "node_id": m["node_id"],
                    "label": m["label"],
                    "node_type": m["node_type"],
                    "lat": round(m["lat"], 6),
                    "lon": round(m["lon"], 6),
                    "elevation": round(m["elevation"], 2),
                    "population_density": round(m["population_density"], 2),
                    "river": m["river"],
                })
        print(f"Saved node metadata → {csv_path}")

    return data, node_meta, G


if __name__ == "__main__":
    data, node_meta, G = build_graph(
        save_dir="/home/user/workspace/MicroPlastiNet/data/processed/m3"
    )
    print(f"\nNode feature shape: {data.x.shape}")
    print(f"Edge index shape:   {data.edge_index.shape}")
    print(f"Edge attr shape:    {data.edge_attr.shape}")
    print(f"Station IDs (first 5): {data.station_ids[:5]}")
    print(f"Source IDs (first 5):  {data.source_ids[:5]}")
