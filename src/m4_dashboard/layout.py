"""
layout.py — MicroPlastiNet M4 Dashboard Layout
Light theme using dash-bootstrap-components (FLATLY).
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

# ─── Color Tokens ──────────────────────────────────────────────────────────────
# Light palette: page is soft slate-50, cards are white, text dark slate.
COLORS = {
    "bg_deep":     "#f8fafc",  # page background
    "bg_panel":    "#ffffff",  # tab strip / panels
    "bg_card":     "#ffffff",  # cards
    "bg_hover":    "#f1f5f9",  # hover surface
    "accent":      "#0f766e",  # primary brand (deep teal)
    "accent_hi":   "#0d9488",  # primary hover
    "accent_lo":   "#ccfbf1",  # primary tint
    "accent_cyan": "#0284c7",  # info / data series
    "accent_teal": "#0f766e",  # legacy alias
    "accent_amber":"#d97706",  # warning
    "accent_red":  "#dc2626",  # danger
    "accent_green":"#16a34a",  # success
    "text_primary":"#0f172a",  # body text
    "text_muted":  "#64748b",  # secondary text
    "border":      "#e2e8f0",  # hairline borders
    "high":        "#dc2626",
    "medium":      "#d97706",
    "low":         "#16a34a",
}

TAB_STYLE = {
    "backgroundColor": COLORS["bg_panel"],
    "color": COLORS["text_muted"],
    "border": f"1px solid {COLORS['border']}",
    "borderBottom": "none",
    "padding": "10px 22px",
    "fontFamily": "'Inter', 'Segoe UI', sans-serif",
    "fontSize": "13px",
    "letterSpacing": "0.03em",
    "fontWeight": "500",
}
TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "backgroundColor": COLORS["bg_card"],
    "color": COLORS["accent"],
    "fontWeight": "600",
    "borderTop": f"2px solid {COLORS['accent']}",
    "borderBottom": f"1px solid {COLORS['bg_card']}",
}

CARD_STYLE = {
    "backgroundColor": COLORS["bg_card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px",
    "padding": "16px",
}

GLOBAL_STYLE = {
    "backgroundColor": COLORS["bg_deep"],
    "minHeight": "100vh",
    "fontFamily": "'Inter', 'Segoe UI', system-ui, sans-serif",
    "color": COLORS["text_primary"],
}


def stat_card(label: str, value: str, color: str = None, icon: str = ""):
    return html.Div(
        [
            html.Div(label, style={"fontSize": "11px", "color": COLORS["text_muted"],
                                   "textTransform": "uppercase", "letterSpacing": "0.08em",
                                   "marginBottom": "4px"}),
            html.Div(
                [html.Span(icon + " " if icon else ""),
                 html.Span(value, style={"fontSize": "22px", "fontWeight": "700",
                                          "color": color or COLORS["accent_cyan"]})],
            ),
        ],
        style={**CARD_STYLE, "minWidth": "120px", "flex": "1"},
    )


def header():
    return html.Div(
        [
            html.Div(
                [
                    # Logo + Title
                    html.Div(
                        [
                            # Logo icon (simple geometric using Div)
                            html.Div(
                                style={
                                    "width": "36px", "height": "27px",
                                    "marginRight": "12px", "flexShrink": "0",
                                    "position": "relative",
                                },
                                children=[
                                    html.Div(style={
                                        "position": "absolute", "left": "6px", "top": "9px",
                                        "width": "10px", "height": "10px",
                                        "borderRadius": "50%",
                                        "backgroundColor": COLORS["accent"],
                                    }),
                                    html.Div(style={
                                        "position": "absolute", "left": "22px", "top": "2px",
                                        "width": "8px", "height": "8px",
                                        "borderRadius": "50%",
                                        "backgroundColor": COLORS["accent_hi"],
                                        "opacity": "0.85",
                                    }),
                                    html.Div(style={
                                        "position": "absolute", "left": "22px", "top": "17px",
                                        "width": "8px", "height": "8px",
                                        "borderRadius": "50%",
                                        "backgroundColor": COLORS["accent_hi"],
                                        "opacity": "0.85",
                                    }),
                                ],
                            ),
                            html.Div(
                                [
                                    html.Span("MicroPlasti", style={"color": COLORS["text_primary"],
                                                                      "fontWeight": "700",
                                                                      "fontSize": "20px"}),
                                    html.Span("Net", style={"color": COLORS["accent"],
                                                             "fontWeight": "700",
                                                             "fontSize": "20px"}),
                                    html.Div("Microplastic Source Attribution · Coastal Georgia Watersheds",
                                             style={"fontSize": "11px", "color": COLORS["text_muted"],
                                                    "marginTop": "1px", "letterSpacing": "0.04em"}),
                                ]
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    # Right-side status indicators (no live indicator — this is a research prototype, not live monitoring)
                    html.Div(
                        [
                            html.Div(
                                children=[
                                    html.Span("DEMO DATASET",
                                              style={"fontSize": "10px",
                                                     "color": COLORS["accent"],
                                                     "letterSpacing": "0.1em",
                                                     "fontWeight": "600",
                                                     "marginRight": "10px",
                                                     "padding": "3px 9px",
                                                     "backgroundColor": COLORS["accent_lo"],
                                                     "borderRadius": "4px"}),
                                ],
                                style={"display": "flex", "alignItems": "center",
                                       "marginRight": "12px"}),
                            html.Div(id="header-clock",
                                     style={"fontSize": "12px", "color": COLORS["text_muted"],
                                            "fontVariantNumeric": "tabular-nums"}),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between",
                       "alignItems": "center"},
            ),
            # Top KPI bar
            html.Div(
                id="kpi-bar",
                style={"display": "flex", "gap": "12px", "marginTop": "16px"},
            ),
        ],
        style={
            "backgroundColor": COLORS["bg_panel"],
            "borderBottom": f"1px solid {COLORS['border']}",
            "padding": "16px 28px",
        },
    )


def tab_live_map():
    return dcc.Tab(
        label="Watershed Map",
        value="tab-map",
        style=TAB_STYLE,
        selected_style=TAB_SELECTED_STYLE,
        children=[
            html.Div(
                [
                    # Left: map
                    html.Div(
                        dcc.Graph(id="map-graph",
                                  config={"displayModeBar": False, "scrollZoom": True},
                                  style={"height": "calc(100vh - 220px)", "minHeight": "520px"}),
                        style={"flex": "1", "minWidth": "0"},
                    ),
                    # Right: station detail panel
                    html.Div(
                        [
                            html.Div("SELECT A STATION", id="station-panel-header",
                                     style={"fontSize": "10px", "color": COLORS["text_muted"],
                                            "letterSpacing": "0.1em", "marginBottom": "14px"}),
                            html.Div(id="station-detail-panel",
                                     children=[
                                         html.Div("Click any station on the map to view details.",
                                                  style={"color": COLORS["text_muted"],
                                                         "fontSize": "13px"}),
                                     ]),
                        ],
                        style={**CARD_STYLE, "width": "280px", "flexShrink": "0",
                               "overflowY": "auto", "maxHeight": "calc(100vh - 220px)"},
                    ),
                ],
                style={"display": "flex", "gap": "12px",
                       "padding": "16px", "alignItems": "flex-start"},
            ),
        ],
    )


def tab_time_series():
    return dcc.Tab(
        label="Station Trends",
        value="tab-ts",
        style=TAB_STYLE,
        selected_style=TAB_SELECTED_STYLE,
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            dbc.Label("Station", style={"fontSize": "12px",
                                                         "color": COLORS["text_muted"],
                                                         "marginBottom": "4px"}),
                            dcc.Dropdown(id="ts-station-select",
                                         placeholder="Select station…",
                                         style={"backgroundColor": COLORS["bg_card"],
                                                "color": "#000"},
                                         clearable=False),
                        ],
                        style={"width": "260px"},
                    ),
                ],
                style={"padding": "16px 16px 0", "display": "flex", "gap": "16px",
                       "alignItems": "flex-end"},
            ),
            html.Div(
                dcc.Graph(id="ts-graph",
                          config={"displayModeBar": "hover"},
                          style={"height": "420px"}),
                style={"padding": "0 16px"},
            ),
            html.Div(id="ts-anomaly-table", style={"padding": "0 16px 16px"}),
        ],
    )


def tab_polymer():
    return dcc.Tab(
        label="Polymer Breakdown",
        value="tab-polymer",
        style=TAB_STYLE,
        selected_style=TAB_SELECTED_STYLE,
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            dbc.Label("Station", style={"fontSize": "12px",
                                                         "color": COLORS["text_muted"]}),
                            dcc.Dropdown(id="poly-station-select",
                                         placeholder="Select station…",
                                         style={"backgroundColor": COLORS["bg_card"],
                                                "color": "#000"},
                                         clearable=False),
                        ],
                        style={"width": "260px"},
                    ),
                ],
                style={"padding": "16px 16px 0"},
            ),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(id="poly-pie", config={"displayModeBar": False},
                                  style={"height": "360px"}),
                        style={**CARD_STYLE, "flex": "1"},
                    ),
                    html.Div(
                        dcc.Graph(id="poly-confidence", config={"displayModeBar": False},
                                  style={"height": "360px"}),
                        style={**CARD_STYLE, "flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "12px", "padding": "16px"},
            ),
            html.Div(
                [
                    html.Div("All Stations — Polymer Stacked Bar",
                             style={"fontSize": "12px", "color": COLORS["text_muted"],
                                    "marginBottom": "8px", "letterSpacing": "0.05em",
                                    "textTransform": "uppercase"}),
                    dcc.Graph(id="poly-stacked-bar", config={"displayModeBar": "hover"},
                              style={"height": "260px"}),
                ],
                style={**CARD_STYLE, "margin": "0 16px 16px"},
            ),
        ],
    )


def tab_source_attribution():
    return dcc.Tab(
        label="Source Attribution",
        value="tab-attr",
        style=TAB_STYLE,
        selected_style=TAB_SELECTED_STYLE,
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            dbc.Label("Station / Contamination Event",
                                      style={"fontSize": "12px", "color": COLORS["text_muted"]}),
                            dcc.Dropdown(id="attr-station-select",
                                         placeholder="Select station…",
                                         style={"backgroundColor": COLORS["bg_card"],
                                                "color": "#000"},
                                         clearable=False),
                        ],
                        style={"width": "300px"},
                    ),
                ],
                style={"padding": "16px 16px 0"},
            ),
            html.Div(
                [
                    # Left: probability bars
                    html.Div(
                        [
                            html.Div("TOP 5 LIKELY SOURCES",
                                     style={"fontSize": "10px", "color": COLORS["text_muted"],
                                            "letterSpacing": "0.1em", "marginBottom": "14px"}),
                            html.Div(id="attr-source-bars"),
                        ],
                        style={**CARD_STYLE, "flex": "1"},
                    ),
                    # Right: attribution map
                    html.Div(
                        [
                            html.Div("HYDROLOGICAL FLOW MAP",
                                     style={"fontSize": "10px", "color": COLORS["text_muted"],
                                            "letterSpacing": "0.1em", "marginBottom": "10px"}),
                            dcc.Graph(id="attr-map", config={"displayModeBar": False},
                                      style={"height": "340px"}),
                        ],
                        style={**CARD_STYLE, "flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "12px", "padding": "16px"},
            ),
            # M3 graph embed
            html.Div(
                [
                    html.Div("GNN GRAPH — SOURCE ATTRIBUTION NETWORK",
                             style={"fontSize": "10px", "color": COLORS["text_muted"],
                                    "letterSpacing": "0.1em", "marginBottom": "10px"}),
                    html.Div(id="m3-graph-embed",
                             children=html.Iframe(
                                 src="/assets/m3_graph.html",
                                 style={
                                     "width": "100%", "height": "420px",
                                     "border": f"1px solid {COLORS['border']}",
                                     "borderRadius": "6px",
                                     "backgroundColor": COLORS["bg_panel"],
                                 }
                             )),
                ],
                style={**CARD_STYLE, "margin": "0 16px 16px"},
            ),
        ],
    )


def tab_predictive():
    return dcc.Tab(
        label="Forecast & Alerts",
        value="tab-predict",
        style=TAB_STYLE,
        selected_style=TAB_SELECTED_STYLE,
        children=[
            html.Div(
                [
                    html.Div("7-DAY FORECAST · All Stations",
                             style={"fontSize": "10px", "color": COLORS["text_muted"],
                                    "letterSpacing": "0.1em", "marginBottom": "6px"}),
                    html.Div(id="alert-station-list",
                             style={"marginBottom": "16px"}),
                ],
                style={"padding": "16px 16px 0"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dbc.Label("Station Forecast Detail",
                                      style={"fontSize": "12px", "color": COLORS["text_muted"]}),
                            dcc.Dropdown(id="forecast-station-select",
                                         placeholder="Select station…",
                                         style={"backgroundColor": COLORS["bg_card"],
                                                "color": "#000"},
                                         clearable=False),
                        ],
                        style={"width": "260px", "marginBottom": "12px"},
                    ),
                    dcc.Graph(id="forecast-graph",
                              config={"displayModeBar": "hover"},
                              style={"height": "380px"}),
                ],
                style={**CARD_STYLE, "margin": "0 16px 16px"},
            ),
        ],
    )


def tab_reports():
    return dcc.Tab(
        label="Reports",
        value="tab-reports",
        style=TAB_STYLE,
        selected_style=TAB_SELECTED_STYLE,
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("GENERATE REGULATOR REPORT",
                                     style={"fontSize": "10px", "color": COLORS["text_muted"],
                                            "letterSpacing": "0.1em", "marginBottom": "14px"}),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dbc.Label("Station",
                                                      style={"fontSize": "12px",
                                                             "color": COLORS["text_muted"]}),
                                            dcc.Dropdown(id="report-station-select",
                                                         placeholder="Select station…",
                                                         style={"backgroundColor": COLORS["bg_card"],
                                                                "color": "#000"},
                                                         clearable=False),
                                        ],
                                        style={"flex": "1"},
                                    ),
                                    html.Div(
                                        [
                                            dbc.Label("Report Mode",
                                                      style={"fontSize": "12px",
                                                             "color": COLORS["text_muted"]}),
                                            dcc.Dropdown(
                                                id="report-mode-select",
                                                options=[
                                                    {"label": "Template (offline)", "value": "template"},
                                                    {"label": "OpenAI GPT-4 (key required)", "value": "openai"},
                                                ],
                                                value="template",
                                                style={"backgroundColor": COLORS["bg_card"],
                                                       "color": "#000"},
                                                clearable=False,
                                            ),
                                        ],
                                        style={"flex": "1"},
                                    ),
                                    html.Button(
                                        "Generate Report",
                                        id="generate-report-btn",
                                        style={
                                            "backgroundColor": COLORS["accent_cyan"],
                                            "color": "#000",
                                            "border": "none",
                                            "borderRadius": "6px",
                                            "padding": "8px 20px",
                                            "fontWeight": "600",
                                            "cursor": "pointer",
                                            "fontSize": "13px",
                                            "alignSelf": "flex-end",
                                            "height": "38px",
                                        },
                                        n_clicks=0,
                                    ),
                                ],
                                style={"display": "flex", "gap": "12px", "alignItems": "flex-end"},
                            ),
                        ],
                        style={**CARD_STYLE, "marginBottom": "16px"},
                    ),
                    # Report display area
                    html.Div(
                        [
                            html.Div("GENERATED REPORT",
                                     style={"fontSize": "10px", "color": COLORS["text_muted"],
                                            "letterSpacing": "0.1em", "marginBottom": "14px",
                                            "display": "flex", "justifyContent": "space-between",
                                            "alignItems": "center"},
                            ),
                            html.Div(id="report-loading-indicator",
                                     style={"display": "none"}),
                            html.Div(
                                id="report-display",
                                children=html.Div(
                                    "Select a station and click Generate Report to produce a regulator-ready document.",
                                    style={"color": COLORS["text_muted"], "fontSize": "13px",
                                           "padding": "40px", "textAlign": "center"}
                                ),
                                style={"backgroundColor": COLORS["bg_panel"],
                                       "border": f"1px solid {COLORS['border']}",
                                       "borderRadius": "6px", "padding": "24px",
                                       "maxHeight": "480px", "overflowY": "auto",
                                       "fontFamily": "'Georgia', 'Times New Roman', serif",
                                       "lineHeight": "1.7"},
                            ),
                            html.Div(
                                [
                                    html.A("Download PDF", id="download-pdf-link",
                                           href="/assets/sample_report.pdf",
                                           target="_blank",
                                           style={"color": COLORS["accent_cyan"],
                                                  "fontSize": "12px",
                                                  "textDecoration": "none",
                                                  "border": f"1px solid {COLORS['accent_cyan']}",
                                                  "padding": "6px 14px",
                                                  "borderRadius": "4px"}),
                                    html.A("Download Markdown", id="download-md-link",
                                           href="/assets/sample_report.md",
                                           target="_blank",
                                           style={"color": COLORS["text_muted"],
                                                  "fontSize": "12px",
                                                  "textDecoration": "none",
                                                  "border": f"1px solid {COLORS['border']}",
                                                  "padding": "6px 14px",
                                                  "borderRadius": "4px"}),
                                ],
                                style={"display": "flex", "gap": "10px", "marginTop": "14px"},
                            ),
                        ],
                        style=CARD_STYLE,
                    ),
                ],
                style={"padding": "16px"},
            ),
        ],
    )


def make_layout(station_options: list) -> html.Div:
    """Build full app layout."""
    return html.Div(
        [
            dcc.Interval(id="clock-interval", interval=1000, n_intervals=0),
            dcc.Store(id="selected-station-store", data=station_options[0]["value"] if station_options else "STN-001"),
            header(),
            dcc.Tabs(
                id="main-tabs",
                value="tab-map",
                style={"backgroundColor": COLORS["bg_panel"],
                       "borderBottom": f"1px solid {COLORS['border']}"},
                children=[
                    tab_live_map(),
                    tab_time_series(),
                    tab_polymer(),
                    tab_source_attribution(),
                    tab_predictive(),
                    tab_reports(),
                ],
            ),
            # Hidden dropdowns seed options from data
            html.Div(id="dummy-output", style={"display": "none"}),
        ],
        style=GLOBAL_STYLE,
    )
