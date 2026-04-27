"""
callbacks.py — MicroPlastiNet M4 Dashboard Callbacks
All Dash interactivity for 6 tabs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback, html, no_update
import dash_bootstrap_components as dbc

from data_loader import (
    load_station_metadata,
    load_time_series,
    load_polymer_breakdown,
    load_source_attribution,
    load_all_polymer_breakdown,
    load_forecast,
    get_map_token,
    POLYMER_TYPES,
    POLYMER_COLORS,
    COLORS as DC,
)

# ─── Plotly Template ───────────────────────────────────────────────────────────
# Base layout - NO margin/xaxis/yaxis/legend here (added per-chart to avoid conflicts)
PLOT_LAYOUT_BASE = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(family="Inter, Segoe UI, system-ui", color="#0f172a", size=12),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor="#0284c7",
                    font=dict(color="#0f172a")),
    colorway=["#0284c7", "#ea580c", "#d97706", "#7c3aed", "#dc2626", "#0d9488"],
)

# Grid axis defaults
AXIS_DEFAULTS = dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", zerolinecolor="#e2e8f0")
PLOT_LAYOUT = PLOT_LAYOUT_BASE  # alias kept for backward compat

STATUS_COLORS = {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a"}

# Cache station metadata
_stations_df = None


def get_stations():
    global _stations_df
    if _stations_df is None:
        _stations_df = load_station_metadata()
    return _stations_df


def register_callbacks(app):
    """Register all callbacks on the Dash app."""

    # ── Clock ──────────────────────────────────────────────────────────────────
    @app.callback(
        Output("header-clock", "children"),
        Input("clock-interval", "n_intervals"),
    )
    def update_clock(n):
        return datetime.now().strftime("%Y-%m-%d  %H:%M:%S UTC")

    # ── KPI Bar (compliance-focused) ───────────────────────────────────────────
    @app.callback(
        Output("kpi-bar", "children"),
        Input("clock-interval", "n_intervals"),
    )
    def update_kpi(n):
        df = get_stations()
        n_high   = int((df["status"] == "HIGH").sum())
        n_medium = int((df["status"] == "MEDIUM").sum())
        n_total  = len(df)
        avg_conc = round(float(df["mp_conc"].mean()), 1)
        max_conc = round(float(df["mp_conc"].max()), 1)
        # Regulatory threshold for freshwater (NOAA Marine Debris Program guidance)
        threshold = 50.0
        n_violation = int((df["mp_conc"] > threshold).sum())
        # Worst watershed by mean concentration
        worst_river = df.groupby("river")["mp_conc"].mean().idxmax() if not df.empty else "—"
        worst_river_conc = round(float(df.groupby("river")["mp_conc"].mean().max()), 1) if not df.empty else 0.0

        def kpi_item(label, value, color="#0f172a", sub="", icon=""):
            return html.Div(
                [
                    html.Div(label, style={"fontSize": "10px", "color": "#64748b",
                                           "textTransform": "uppercase",
                                           "fontWeight": "600",
                                           "letterSpacing": "0.08em", "marginBottom": "6px"}),
                    html.Div(
                        [html.Span(icon, style={"marginRight": "6px", "fontSize": "11px",
                                                 "color": color, "verticalAlign": "middle"}),
                         html.Span(str(value), style={"fontWeight": "700", "fontSize": "22px",
                                                       "color": color,
                                                       "fontFamily": "'JetBrains Mono', monospace",
                                                       "letterSpacing": "-0.02em"})],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.Div(sub, style={"fontSize": "11px", "color": "#64748b",
                                          "marginTop": "3px"}) if sub else None,
                ],
                style={"backgroundColor": "#ffffff", "border": "1px solid #e2e8f0",
                       "borderRadius": "8px", "padding": "12px 16px", "flex": "1",
                       "minWidth": "140px"},
            )

        return [
            kpi_item("Stations in Violation", n_violation, "#dc2626",
                     sub=f"of {n_total} · threshold {int(threshold)} p/L", icon="●"),
            kpi_item("High-Risk Stations", n_high, "#dc2626",
                     sub=f"medium: {n_medium} · high+med: {n_high + n_medium}"),
            kpi_item("Peak Concentration", f"{max_conc:.1f}", "#d97706",
                     sub="particles/L · worst station"),
            kpi_item("Network Average", f"{avg_conc:.1f}", "#0f766e",
                     sub=f"particles/L across {n_total} stations"),
            kpi_item("Highest-Risk Watershed", worst_river, "#0f766e",
                     sub=f"mean {worst_river_conc:.1f} p/L"),
        ]

    # ── Map ────────────────────────────────────────────────────────────────────
    @app.callback(
        Output("map-graph", "figure"),
        Input("clock-interval", "n_intervals"),
    )
    def update_map(n):
        df = get_stations()
        fig = go.Figure()

        for status, color in STATUS_COLORS.items():
            sub = df[df["status"] == status]
            fig.add_trace(go.Scattermap(
                lat=sub["lat"], lon=sub["lon"],
                mode="markers",
                marker=dict(
                    size=12 if status == "HIGH" else (10 if status == "MEDIUM" else 9),
                    color=color,
                    opacity=0.9,
                    symbol="circle",
                ),
                name=f"{status} ({len(sub)})",
                text=sub["name"],
                customdata=sub[["station_id", "mp_conc", "river", "turbidity_ntu", "ph"]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Station: %{customdata[0]}<br>"
                    "Conc: %{customdata[1]:.1f} p/L<br>"
                    "River: %{customdata[2]}<br>"
                    "Turbidity: %{customdata[3]:.1f} NTU<br>"
                    "pH: %{customdata[4]:.2f}<extra></extra>"
                ),
            ))

        fig.update_layout(
            **PLOT_LAYOUT_BASE,
            map=dict(
                style="open-street-map",
                center=dict(lat=32.1, lon=-81.4),
                zoom=7.5,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#e2e8f0",
                borderwidth=1,
                font=dict(color="#0f172a", size=11),
                x=0.01, y=0.99,
            ),
            showlegend=True,
        )
        return fig

    # ── Map click → Station detail panel ──────────────────────────────────────
    @app.callback(
        [Output("station-detail-panel", "children"),
         Output("station-panel-header", "children"),
         Output("selected-station-store", "data")],
        Input("map-graph", "clickData"),
        State("selected-station-store", "data"),
    )
    def update_station_panel(click_data, current_station):
        df = get_stations()
        if click_data is None:
            station_id = current_station or df["station_id"].iloc[0]
        else:
            pt = click_data["points"][0]
            station_id = pt["customdata"][0]

        row = df[df["station_id"] == station_id].iloc[0]
        pb  = load_polymer_breakdown(station_id)
        top_polymer = max(pb["polymers"], key=pb["polymers"].get)
        top_pct     = round(pb["polymers"][top_polymer] * 100, 1)

        status_color = STATUS_COLORS.get(row["status"], "#94a3b8")

        def detail_row(label, value, color=None):
            return html.Div(
                [
                    html.Span(label, style={"fontSize": "11px", "color": "#64748b",
                                            "width": "110px", "display": "inline-block"}),
                    html.Span(str(value), style={"fontSize": "12px", "color": color or "#0f172a",
                                                 "fontWeight": "500"}),
                ],
                style={"marginBottom": "8px"},
            )

        return (
            [
                html.Div(
                    [
                        html.Span(row["status"],
                                  style={"fontSize": "10px", "fontWeight": "700",
                                         "color": status_color,
                                         "border": f"1px solid {status_color}",
                                         "borderRadius": "3px", "padding": "2px 8px",
                                         "letterSpacing": "0.08em"}),
                    ],
                    style={"marginBottom": "14px"},
                ),
                detail_row("Station ID", station_id),
                detail_row("River", row["river"]),
                detail_row("MP Conc", f"{row['mp_conc']:.1f} p/L", status_color),
                detail_row("Turbidity", f"{row['turbidity_ntu']:.1f} NTU"),
                detail_row("pH", f"{row['ph']:.2f}"),
                detail_row("Temp", f"{row['temp_c']:.1f} °C"),
                detail_row("Depth", f"{row['depth_m']:.1f} m"),
                html.Hr(style={"borderColor": "#e2e8f0", "margin": "10px 0"}),
                detail_row("Top Polymer", f"{top_polymer} ({top_pct}%)"),
                detail_row("Total Particles", f"{pb['total_particles']:,}"),
                detail_row("Lat/Lon", f"{row['lat']:.4f}, {row['lon']:.4f}"),
                detail_row("Installed", row["install_date"]),
            ],
            f"STATION — {station_id}",
            station_id,
        )

    # ── Time Series ────────────────────────────────────────────────────────────
    @app.callback(
        [Output("ts-station-select", "options"),
         Output("ts-station-select", "value")],
        Input("clock-interval", "n_intervals"),
    )
    def populate_ts_dropdown(n):
        df = get_stations()
        opts = [{"label": f"{r['station_id']} — {r['name']}", "value": r["station_id"]}
                for _, r in df.iterrows()]
        return opts, opts[0]["value"]

    @app.callback(
        [Output("ts-graph", "figure"),
         Output("ts-anomaly-table", "children")],
        Input("ts-station-select", "value"),
    )
    def update_ts(station_id):
        if not station_id:
            return go.Figure(), html.Div()
        ts = load_time_series(station_id, days=30)
        fig = go.Figure()

        # Main line
        fig.add_trace(go.Scatter(
            x=ts["date"], y=ts["mp_conc"],
            mode="lines",
            name="MP Concentration",
            line=dict(color="#0284c7", width=2),
            fill="tozeroy",
            fillcolor="rgba(2,132,199,0.10)",
            hovertemplate="%{x|%b %d}<br>%{y:.1f} p/L<extra></extra>",
        ))

        # Anomaly markers
        anomalies = ts[ts["anomaly"]]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies["date"], y=anomalies["mp_conc"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#dc2626", size=10, symbol="diamond",
                             line=dict(color="#ffffff", width=1)),
                hovertemplate="%{x|%b %d}<br><b>ANOMALY: %{y:.1f} p/L</b><extra></extra>",
            ))

        # Turbidity secondary axis
        fig.add_trace(go.Scatter(
            x=ts["date"], y=ts["turbidity"],
            mode="lines",
            name="Turbidity (NTU)",
            line=dict(color="#d97706", width=1.5, dash="dot"),
            yaxis="y2",
            hovertemplate="%{x|%b %d}<br>%{y:.1f} NTU<extra></extra>",
        ))

        fig.update_layout(
            **PLOT_LAYOUT_BASE,
            margin=dict(l=55, r=55, t=45, b=40),
            title=dict(text=f"{station_id} — 30-Day MP Concentration",
                        font=dict(size=13, color="#64748b"), x=0),
            xaxis=dict(**AXIS_DEFAULTS),
            yaxis=dict(**AXIS_DEFAULTS, title="MP Concentration (p/L)"),
            yaxis2=dict(overlaying="y", side="right",
                        title=dict(text="Turbidity (NTU)", font=dict(color="#d97706")),
                        gridcolor="#e2e8f0", linecolor="#cbd5e1"),
            hovermode="x unified",
        )

        # Anomaly table
        if anomalies.empty:
            table = html.Div("No anomalies detected in the past 30 days.",
                             style={"color": "#16a34a", "fontSize": "12px",
                                    "padding": "8px 0"})
        else:
            rows = [
                html.Tr([
                    html.Td(row["date"].strftime("%Y-%m-%d"),
                            style={"padding": "6px 12px", "color": "#64748b", "fontSize": "12px"}),
                    html.Td(f"{row['mp_conc']:.1f} p/L",
                            style={"padding": "6px 12px", "color": "#dc2626",
                                   "fontWeight": "600", "fontSize": "12px"}),
                    html.Td("⚠ Spike Detected",
                            style={"padding": "6px 12px", "color": "#d97706", "fontSize": "12px"}),
                ])
                for _, row in anomalies.iterrows()
            ]
            table = html.Div(
                [
                    html.Div("ANOMALY LOG", style={"fontSize": "10px", "color": "#64748b",
                                                     "letterSpacing": "0.1em",
                                                     "marginBottom": "8px"}),
                    html.Table(
                        [html.Thead(html.Tr([
                            html.Th("Date", style={"padding": "6px 12px", "fontSize": "11px",
                                                    "color": "#64748b", "fontWeight": "600"}),
                            html.Th("Concentration", style={"padding": "6px 12px",
                                                             "fontSize": "11px",
                                                             "color": "#64748b",
                                                             "fontWeight": "600"}),
                            html.Th("Flag", style={"padding": "6px 12px", "fontSize": "11px",
                                                    "color": "#64748b", "fontWeight": "600"}),
                        ]))] + [html.Tbody(rows)],
                        style={"width": "100%", "borderCollapse": "collapse",
                               "backgroundColor": "#ffffff",
                               "border": "1px solid #e2e8f0", "borderRadius": "6px"},
                    ),
                ],
                style={"marginTop": "8px"},
            )
        return fig, table

    # ── Polymer Breakdown ──────────────────────────────────────────────────────
    @app.callback(
        [Output("poly-station-select", "options"),
         Output("poly-station-select", "value")],
        Input("clock-interval", "n_intervals"),
    )
    def populate_poly_dropdown(n):
        df = get_stations()
        opts = [{"label": f"{r['station_id']} — {r['name']}", "value": r["station_id"]}
                for _, r in df.iterrows()]
        return opts, opts[0]["value"]

    @app.callback(
        [Output("poly-pie", "figure"),
         Output("poly-confidence", "figure"),
         Output("poly-stacked-bar", "figure")],
        Input("poly-station-select", "value"),
    )
    def update_polymer(station_id):
        if not station_id:
            empty = go.Figure()
            empty.update_layout(**PLOT_LAYOUT_BASE)
            return empty, empty, empty

        pb = load_polymer_breakdown(station_id)
        polymers = pb["polymers"]
        confidence = pb["confidence"]

        # Pie chart
        labels = list(polymers.keys())
        values = [polymers[p] * 100 for p in labels]
        colors = [POLYMER_COLORS[p] for p in labels]

        pie_fig = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
            hole=0.42,
            hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
            textinfo="label+percent",
            textfont=dict(size=12, color="#0f172a"),
        ))
        pie_fig.update_layout(
            **PLOT_LAYOUT_BASE,
            margin=dict(l=20, r=20, t=45, b=20),
            title=dict(text=f"{station_id} — Polymer Composition",
                        font=dict(size=13, color="#64748b"), x=0),
            annotations=[dict(text=f"{pb['total_particles']:,}<br>particles",
                               x=0.5, y=0.5, font_size=13, showarrow=False,
                               font=dict(color="#0f172a"))],
        )

        # Confidence bar chart
        sorted_poly = sorted(confidence.items(), key=lambda x: x[1], reverse=True)
        bar_fig = go.Figure(go.Bar(
            x=[v * 100 for _, v in sorted_poly],
            y=[p for p, _ in sorted_poly],
            orientation="h",
            marker=dict(
                color=[POLYMER_COLORS[p] for p, _ in sorted_poly],
                opacity=0.85,
            ),
            hovertemplate="<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>",
            text=[f"{v*100:.0f}%" for _, v in sorted_poly],
            textposition="outside",
            textfont=dict(color="#0f172a", size=11),
        ))
        bar_fig.update_layout(
            **PLOT_LAYOUT_BASE,
            margin=dict(l=80, r=20, t=45, b=40),
            title=dict(text="Classifier Confidence by Polymer",
                        font=dict(size=13, color="#64748b"), x=0),
            xaxis=dict(**AXIS_DEFAULTS, title="Confidence (%)", range=[0, 110]),
            yaxis=dict(**AXIS_DEFAULTS, title=""),
        )

        # Stacked bar — all stations
        all_df = load_all_polymer_breakdown()
        stacked_fig = go.Figure()
        for polymer in POLYMER_TYPES:
            stacked_fig.add_trace(go.Bar(
                x=all_df["station_id"],
                y=all_df[polymer] * 100,
                name=polymer,
                marker_color=POLYMER_COLORS[polymer],
                hovertemplate=f"<b>{polymer}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
            ))
        stacked_fig.update_layout(
            **PLOT_LAYOUT_BASE,
            margin=dict(l=50, r=20, t=45, b=80),
            barmode="stack",
            title=dict(text="Polymer Distribution — All Stations",
                        font=dict(size=13, color="#64748b"), x=0),
            xaxis=dict(**AXIS_DEFAULTS, tickangle=-60, tickfont=dict(size=9)),
            yaxis=dict(**AXIS_DEFAULTS, title="Proportion (%)"),
        )
        return pie_fig, bar_fig, stacked_fig

    # ── Source Attribution ─────────────────────────────────────────────────────
    @app.callback(
        [Output("attr-station-select", "options"),
         Output("attr-station-select", "value")],
        Input("clock-interval", "n_intervals"),
    )
    def populate_attr_dropdown(n):
        df = get_stations()
        opts = [{"label": f"{r['station_id']} — {r['name']}", "value": r["station_id"]}
                for _, r in df.iterrows()]
        return opts, opts[0]["value"]

    @app.callback(
        [Output("attr-source-bars", "children"),
         Output("attr-map", "figure")],
        Input("attr-station-select", "value"),
    )
    def update_attribution(station_id):
        if not station_id:
            return html.Div(), go.Figure()

        df = get_stations()
        attr = load_source_attribution(station_id)
        station_row = df[df["station_id"] == station_id].iloc[0]

        # Source probability bars
        source_bars = []
        for src in attr["sources"]:
            pct = src["probability"] * 100
            conf = src["confidence"] * 100
            bar = html.Div(
                [
                    html.Div(
                        [
                            html.Span(f"#{src['rank']} {src['name']}",
                                      style={"fontSize": "13px", "fontWeight": "500",
                                             "color": "#0f172a"}),
                            html.Span(f"{pct:.1f}%",
                                      style={"fontSize": "13px", "fontWeight": "700",
                                             "color": "#0284c7"}),
                        ],
                        style={"display": "flex", "justifyContent": "space-between",
                               "marginBottom": "4px"},
                    ),
                    html.Div(
                        html.Div(style={
                            "width": f"{min(pct, 100)}%",
                            "height": "8px",
                            "backgroundColor": "#0284c7",
                            "borderRadius": "4px",
                            "opacity": "0.85",
                            "transition": "width 0.5s ease",
                        }),
                        style={"backgroundColor": "#e2e8f0", "borderRadius": "4px",
                               "height": "8px", "marginBottom": "4px"},
                    ),
                    html.Div(
                        [
                            html.Span(f"Confidence: {conf:.0f}%  ·  ",
                                      style={"fontSize": "11px", "color": "#64748b"}),
                            html.Span(f"Distance: {src['distance_km']} km",
                                      style={"fontSize": "11px", "color": "#64748b"}),
                        ],
                    ),
                ],
                style={"marginBottom": "18px"},
            )
            source_bars.append(bar)

        event_info = html.Div(
            [
                html.Div(f"Event ID: {attr['event_id']}  ·  Date: {attr['event_date']}",
                         style={"fontSize": "11px", "color": "#64748b",
                                "marginBottom": "16px",
                                "padding": "6px 10px",
                                "border": "1px solid #e2e8f0",
                                "borderRadius": "4px"}),
            ]
        )

        # Attribution map
        map_fig = go.Figure()

        # Add station marker
        map_fig.add_trace(go.Scattermap(
            lat=[station_row["lat"]], lon=[station_row["lon"]],
            mode="markers+text",
            marker=dict(size=16, color="#0284c7", symbol="circle"),
            text=[station_id],
            textposition="top right",
            textfont=dict(color="#0284c7", size=11),
            name="Monitoring Station",
            hovertemplate=f"<b>{station_id}</b><br>Detection site<extra></extra>",
        ))

        # Add source markers with lines
        colors_sources = ["#dc2626", "#d97706", "#7c3aed", "#0d9488", "#ea580c"]
        for i, src in enumerate(attr["sources"]):
            c = colors_sources[i % len(colors_sources)]
            # Line from source to station
            map_fig.add_trace(go.Scattermap(
                lat=[src["lat"], station_row["lat"]],
                lon=[src["lon"], station_row["lon"]],
                mode="lines",
                line=dict(width=1.5, color=c),
                opacity=0.4,
                showlegend=False,
                hoverinfo="skip",
            ))
            map_fig.add_trace(go.Scattermap(
                lat=[src["lat"]], lon=[src["lon"]],
                mode="markers",
                marker=dict(size=10 - i, color=c, symbol="circle"),
                name=f"#{src['rank']} {src['name'][:20]}",
                hovertemplate=(
                    f"<b>#{src['rank']} {src['name']}</b><br>"
                    f"Prob: {src['probability']*100:.1f}%<br>"
                    f"Dist: {src['distance_km']} km<extra></extra>"
                ),
            ))

        map_fig.update_layout(
            **PLOT_LAYOUT_BASE,
            map=dict(
                style="open-street-map",
                center=dict(lat=float(station_row["lat"]),
                            lon=float(station_row["lon"])),
                zoom=8,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#e2e8f0",
                        borderwidth=1, font=dict(color="#0f172a", size=10),
                        x=0.01, y=0.99),
        )

        return [event_info] + source_bars, map_fig

    # ── Predictive Alerts ──────────────────────────────────────────────────────
    @app.callback(
        [Output("forecast-station-select", "options"),
         Output("forecast-station-select", "value"),
         Output("alert-station-list", "children")],
        Input("clock-interval", "n_intervals"),
    )
    def populate_forecast(n):
        df = get_stations()
        opts = [{"label": f"{r['station_id']} — {r['name']}", "value": r["station_id"]}
                for _, r in df.iterrows()]

        # Show HIGH-status stations as alert cards
        high_stations = df[df["status"] == "HIGH"].head(8)
        alert_cards = []
        for _, row in high_stations.iterrows():
            alert_cards.append(html.Div(
                [
                    html.Span("⚠", style={"color": "#dc2626", "marginRight": "6px",
                                          "fontSize": "14px"}),
                    html.Span(row["station_id"],
                              style={"fontWeight": "600", "color": "#0f172a",
                                     "fontSize": "12px", "marginRight": "8px"}),
                    html.Span(f"{row['mp_conc']:.1f} p/L",
                              style={"color": "#dc2626", "fontSize": "12px",
                                     "fontWeight": "500", "marginRight": "6px"}),
                    html.Span(row["river"],
                              style={"color": "#64748b", "fontSize": "11px"}),
                ],
                style={"display": "inline-flex", "alignItems": "center",
                       "backgroundColor": "rgba(220,38,38,0.08)",
                       "border": "1px solid rgba(220,38,38,0.3)",
                       "borderRadius": "4px", "padding": "4px 10px",
                       "marginRight": "8px", "marginBottom": "6px"},
            ))

        alert_list = html.Div(
            [html.Div("HIGH ALERT STATIONS", style={"fontSize": "10px", "color": "#64748b",
                                                      "letterSpacing": "0.1em",
                                                      "marginBottom": "8px"})] + alert_cards
        )
        return opts, opts[0]["value"], alert_list

    @app.callback(
        Output("forecast-graph", "figure"),
        Input("forecast-station-select", "value"),
    )
    def update_forecast(station_id):
        if not station_id:
            return go.Figure()

        ts  = load_time_series(station_id, days=14)
        fc  = load_forecast(station_id, days_ahead=7)

        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(
            x=ts["date"], y=ts["mp_conc"],
            mode="lines",
            name="Historical",
            line=dict(color="#0284c7", width=2),
            hovertemplate="%{x|%b %d}<br>%{y:.1f} p/L<extra></extra>",
        ))

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([fc["date"], fc["date"][::-1]]),
            y=pd.concat([fc["upper_ci"], fc["lower_ci"][::-1]]),
            fill="toself",
            fillcolor="rgba(255,107,53,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% CI",
            showlegend=True,
            hoverinfo="skip",
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=fc["date"], y=fc["predicted"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#ea580c", width=2, dash="dash"),
            marker=dict(
                color=["#dc2626" if a else "#ea580c" for a in fc["alert"]],
                size=8,
                line=dict(color="#ffffff", width=1),
            ),
            hovertemplate="%{x|%b %d}<br>Forecast: %{y:.1f} p/L<extra></extra>",
        ))

        # Alert threshold line
        alert_threshold = 65.0
        fig.add_hline(y=alert_threshold, line_dash="dot",
                       line_color="#dc2626", line_width=1.5,
                       annotation_text="Alert Threshold (65 p/L)",
                       annotation_font=dict(color="#dc2626", size=11),
                       annotation_position="top left")

        fig.update_layout(
            **PLOT_LAYOUT_BASE,
            margin=dict(l=55, r=20, t=45, b=40),
            title=dict(text=f"{station_id} — 7-Day Forecast + Historical",
                        font=dict(size=13, color="#64748b"), x=0),
            xaxis=dict(**AXIS_DEFAULTS),
            yaxis=dict(**AXIS_DEFAULTS, title="MP Concentration (p/L)"),
            hovermode="x unified",
        )
        return fig

    # ── Reports ────────────────────────────────────────────────────────────────
    @app.callback(
        [Output("report-station-select", "options"),
         Output("report-station-select", "value")],
        Input("clock-interval", "n_intervals"),
    )
    def populate_report_dropdown(n):
        df = get_stations()
        opts = [{"label": f"{r['station_id']} — {r['name']}", "value": r["station_id"]}
                for _, r in df.iterrows()]
        return opts, opts[0]["value"]

    @app.callback(
        Output("report-display", "children"),
        Input("generate-report-btn", "n_clicks"),
        State("report-station-select", "value"),
        State("report-mode-select", "value"),
        prevent_initial_call=True,
    )
    def generate_report_display(n_clicks, station_id, mode):
        if not station_id or n_clicks == 0:
            return no_update

        try:
            import importlib.util
            m5_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "m5_genai"))
            if m5_dir not in sys.path:
                sys.path.insert(0, m5_dir)
            m5_path = os.path.join(m5_dir, "report_generator.py")
            spec = importlib.util.spec_from_file_location("report_generator", m5_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            event_data   = load_polymer_breakdown(station_id)
            attr_data    = load_source_attribution(station_id)
            report_text  = mod.generate_report(station_id, event_data, attr_data, mode=mode)

            # Render markdown-like text in HTML
            paragraphs = report_text.split("\n\n")
            elements = []
            for para in paragraphs:
                if para.startswith("# "):
                    elements.append(html.H2(para[2:], style={"color": "#0284c7",
                                                               "fontSize": "18px",
                                                               "borderBottom": "1px solid #e2e8f0",
                                                               "paddingBottom": "8px",
                                                               "marginTop": "20px"}))
                elif para.startswith("## "):
                    elements.append(html.H3(para[3:], style={"color": "#0f172a",
                                                               "fontSize": "14px",
                                                               "fontWeight": "600",
                                                               "marginTop": "16px"}))
                elif para.startswith("**") and para.endswith("**"):
                    elements.append(html.P(para[2:-2], style={"fontWeight": "600",
                                                               "color": "#0f172a"}))
                else:
                    lines = para.split("\n")
                    for line in lines:
                        if line.startswith("- "):
                            elements.append(html.Li(line[2:], style={"color": "#0f172a",
                                                                       "marginBottom": "4px"}))
                        elif line.strip():
                            elements.append(html.P(line, style={"color": "#0f172a",
                                                                 "marginBottom": "8px",
                                                                 "lineHeight": "1.7"}))
            return elements

        except Exception as e:
            return html.Div(
                f"Error generating report: {str(e)}",
                style={"color": "#dc2626", "fontSize": "13px"},
            )
