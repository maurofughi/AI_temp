# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 13:00:40 2025

@author: mauro

pages/page5.py
Phase 3 - ML CPO (FWA runner + outputs)
"""

import os
import json
from datetime import datetime

import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, ctx
from dash.dependencies import Input, Output, State
from dash import dash_table
import plotly.graph_objects as go


from ml.ml1 import RunParams, run_fwa_single
from ml.ml2 import RunParamsWeekly, run_fwa_weekly



# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # .../Portfolio26
ML_DIR = os.path.join(BASE_DIR, "ml")
ML_DATASETS_JSON = os.path.join(ML_DIR, "ml_datasets.json")


def _load_ml_datasets_index() -> dict:
    if not os.path.exists(ML_DATASETS_JSON):
        return {"datasets": []}
    with open(ML_DATASETS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _dataset_options_from_index(idx: dict) -> list:
    rows = idx.get("datasets", []) or []

    # keep only snapshots that actually have a saved CSV path
    valid = []
    for r in rows:
        ds = r.get("dataset") or {}
        path = ds.get("path")
        saved = ds.get("saved", False)
        if isinstance(path, str) and path.strip() and saved is True:
            valid.append(r)

    # newest first (snapshot_id is sortable; created_at also works)
    valid.sort(key=lambda r: r.get("created_at", ""), reverse=True)

    opts = []
    for r in valid:
        ds = r.get("dataset") or {}
        path = ds["path"]  # guaranteed by filter above

        snap = r.get("snapshot_id", "")
        n_strat = ds.get("n_strategies", "")
        n_rows = ds.get("n_rows", "")
        min_dt = ds.get("min_open_dt", "")
        max_dt = ds.get("max_open_dt", "")

        label = f"{snap} | N={n_strat} | rows={n_rows} | {str(min_dt)[:10]}→{str(max_dt)[:10]}"
        opts.append({"label": label, "value": path})

    return opts



# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
def build_ml_cpo_right_panel():
    return dbc.Container(
        [
            html.H3("ML CPO", className="mb-2"),
            dcc.Store(id="mlcpo-curves-store", data=None),
            dbc.Row(
                [
                    # Left/top: parameters (tight)
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("ML Run Parameters"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("ML Dataset (snapshot)"),
                                                        dcc.Dropdown(
                                                            id="mlcpo-dataset-dropdown",
                                                            options=[],
                                                            value=None,
                                                            placeholder="Select a built ML dataset (from ML Utilities)…",
                                                            className="p26-dark-dropdown",
                                                            style={"fontSize": "0.85rem"},
                                                        ),
                                                    ],
                                                    width=10,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Button(
                                                            "Refresh",
                                                            id="mlcpo-refresh-datasets-btn",
                                                            size="sm",
                                                            color="secondary",
                                                            outline=True,
                                                            style={"marginTop": "1.75rem", "width": "100%"},
                                                        )
                                                    ],
                                                    width=2,
                                                ),
                                            ],
                                            className="g-2",
                                        ),
                                        #here
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.ButtonGroup(
                                                        [
                                                            dbc.Button(
                                                                "Monthly (Static)",
                                                                id="mlcpo-mode-monthly-btn",
                                                                n_clicks=0,
                                                                color="primary",
                                                                size="sm",
                                                            ),
                                                            dbc.Button(
                                                                "Weekly (CPO)",
                                                                id="mlcpo-mode-weekly-btn",
                                                                n_clicks=0,
                                                                color="secondary",
                                                                size="sm",
                                                            ),
                                                        ]
                                                    ),
                                                    width="auto",
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        id="mlcpo-mode-hint",
                                                        children="Mode: Monthly (Static)",
                                                        style={"color": "rgba(255,255,255,0.7)", "paddingTop": "6px"},
                                                    )
                                                ),
                                            ],
                                            className="g-2",
                                            style={"marginTop": "6px", "marginBottom": "6px"},
                                        ),

                                        html.Hr(className="my-2"),

                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Start Date override (optional)"),
                                                        dbc.Input(
                                                            id="mlcpo-start-date",
                                                            type="text",
                                                            placeholder="YYYY-MM-DD (blank = data min)",
                                                            value="",                                                           
                                                            size="sm",
                                                            style={
                                                                "backgroundColor": "#1e1e1e",
                                                                "color": "white",
                                                                "border": "1px solid #444",
                                                            }
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("End Date override (optional)"),
                                                        dbc.Input(
                                                            id="mlcpo-end-date",
                                                            type="text",
                                                            placeholder="YYYY-MM-DD (blank = data max)",
                                                            value="",
                                                            size="sm",
                                                            style={
                                                                "backgroundColor": "#1e1e1e",
                                                                "color": "white",
                                                                "border": "1px solid #444",
                                                            }
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ],
                                            className="g-2",
                                        ),
                                        # ---- IS + Anchored (always visible) ----
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("IS months"),
                                                        dbc.Input(
                                                            id="mlcpo-is-months",
                                                            type="number",
                                                            value=2,
                                                            min=1,
                                                            step=1,
                                                            size="sm",
                                                            style={
                                                                "backgroundColor": "#1e1e1e",
                                                                "color": "white",
                                                                "border": "1px solid #444",
                                                            },
                                                        ),
                                                    ],
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Anchored"),
                                                        dbc.RadioItems(
                                                            id="mlcpo-anchored-type",
                                                            options=[
                                                                {"label": "Unanchored", "value": "U"},
                                                                {"label": "Anchored", "value": "A"},
                                                            ],
                                                            value="U",
                                                            inline=True,
                                                            style={"fontSize": "0.85rem"},
                                                        ),
                                                    ],
                                                    width=9,
                                                ),
                                            ],
                                            className="g-2 mt-1",
                                        ),
                                        
                                        # ---- Monthly params container (shown in Monthly mode) ----
                                        html.Div(
                                            id="mlcpo-monthly-params",
                                            style={"display": "block"},
                                            children=[
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("OoS months"),
                                                                dbc.Input(
                                                                    id="mlcpo-oos-months",
                                                                    type="number",
                                                                    value=1,
                                                                    min=1,
                                                                    step=1,
                                                                    size="sm",
                                                                    style={
                                                                        "backgroundColor": "#1e1e1e",
                                                                        "color": "white",
                                                                        "border": "1px solid #444",
                                                                    },
                                                                ),
                                                            ],
                                                            width=3,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Step months"),
                                                                dbc.Input(
                                                                    id="mlcpo-step-months",
                                                                    type="number",
                                                                    value=1,
                                                                    min=1,
                                                                    step=1,
                                                                    size="sm",
                                                                    style={
                                                                        "backgroundColor": "#1e1e1e",
                                                                        "color": "white",
                                                                        "border": "1px solid #444",
                                                                    },
                                                                ),
                                                            ],
                                                            width=3,
                                                        ),
                                                    ],
                                                    className="g-2 mt-1",
                                                )
                                            ],
                                        ),
                                        
                                        # ---- Weekly params container (shown in Weekly mode) ----
                                        html.Div(
                                            id="mlcpo-weekly-params",
                                            style={"display": "none"},
                                            children=[
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("OoS weeks"),
                                                                dbc.Input(
                                                                    id="mlcpo-oos-weeks",
                                                                    type="number",
                                                                    value=1,
                                                                    min=1,
                                                                    step=1,
                                                                    size="sm",
                                                                    style={
                                                                        "backgroundColor": "#1e1e1e",
                                                                        "color": "white",
                                                                        "border": "1px solid #444",
                                                                    },
                                                                ),
                                                            ],
                                                            width=3,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Step weeks"),
                                                                dbc.Input(
                                                                    id="mlcpo-step-weeks",
                                                                    type="number",
                                                                    value=1,
                                                                    min=1,
                                                                    step=1,
                                                                    size="sm",
                                                                    style={
                                                                        "backgroundColor": "#1e1e1e",
                                                                        "color": "white",
                                                                        "border": "1px solid #444",
                                                                    },
                                                                ),
                                                            ],
                                                            width=3,
                                                        ),
                                                    ],
                                                    className="g-2 mt-1",
                                                )
                                            ],
                                        ),

                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Selection mode"),
                                                        dcc.Dropdown(
                                                            id="mlcpo-selection-mode",
                                                            options=[
                                                                {
                                                                    "label": "Top-K per day (by predicted prob)",
                                                                    "value": "topk_per_day",
                                                                },
                                                                {
                                                                    "label": "Bottom-K per day (by predicted prob)",
                                                                    "value": "bottomk_per_day",
                                                                },
                                                                {
                                                                    "label": "Bottom-p percentile (remove worst p% by ML rank)",
                                                                    "value": "bottomp_perc",
                                                                },
                                                            ],
                                                            value="topk_per_day",
                                                            className="p26-dark-dropdown",
                                                            clearable=False,
                                                            style={"fontSize": "0.85rem"},
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        # Label text will be updated dynamically by a callback
                                                        dbc.Label(
                                                            id="mlcpo-selection-param-label",
                                                            children="Top K",
                                                        ),
                                                        dbc.Input(
                                                            id="mlcpo-top-k",
                                                            type="number",
                                                            value=3,
                                                            min=1,
                                                            step=1,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Button(
                                                            "Run ML (FWA)",
                                                            id="mlcpo-run-btn",
                                                            color="primary",
                                                            size="sm",
                                                            style={"marginTop": "1.75rem", "width": "100%"},
                                                        )
                                                    ],
                                                    width=2,
                                                ),
                                            ],
                                            className="g-2 mt-1",
                                        ),
                                        #here
                                        dbc.Button(
                                            "Charts",
                                            id="mlcpo-open-charts-btn",
                                            color="secondary",
                                            outline=True,
                                            size="sm",
                                            style={"marginTop": "1.75rem", "width": "100%"},
                                        ),                                        
                                       
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        width=6,
                    ),

                    # Right/top: summary box
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Run Summary"),
                                dbc.CardBody(
                                    dbc.Spinner(
                                        [
                                            dbc.Alert(
                                                "No run yet.",
                                                id="mlcpo-status",
                                                color="secondary",
                                                className="mb-2",
                                            ),
                                            html.Pre(
                                                "",
                                                id="mlcpo-summary-text",
                                                style={
                                                    "whiteSpace": "pre-wrap",
                                                    "fontSize": "0.85rem",
                                                    "marginBottom": "0",
                                                    "minHeight": "120px",
                                                },
                                            ),
                                        ],
                                        size="sm",
                                        color="primary",
                                        fullscreen=False,
                                    )
                                ),

                            ],
                            className="mb-3",
                        ),
                        width=6,
                    ),
                ],
                className="g-3",
            ),
            # Before/After metrics
            dbc.Card(
                [
                    dbc.CardHeader("Baseline vs ML Metrics (OoS candidates vs Selected)"),
                    dbc.CardBody(
                        [
                            dash_table.DataTable(
                                id="mlcpo-metrics-table",
                                columns=[],
                                data=[],
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "fontSize": "0.85rem",
                                    "padding": "6px",
                                    "whiteSpace": "nowrap",
                                    "backgroundColor": "rgba(0,0,0,0)",
                                    "color": "white",
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "rgba(255,255,255,0.08)",
                                    "color": "white",
                                },
                                css=[
                                    # Row hover -> blue highlight instead of white
                                    {
                                        "selector": ".dash-spreadsheet-container .dash-spreadsheet-inner tr:hover td",
                                        "rule": "background-color: rgba(0, 0, 255, 0.7) !important; color: white !important;",
                                    },
                                    # Also keep active/selected readable (click focus)
                                    {
                                        "selector": ".dash-spreadsheet-container .dash-spreadsheet-inner td.focused, "
                                                    ".dash-spreadsheet-container .dash-spreadsheet-inner td.cell--selected",
                                        "rule": "background-color: rgba(0, 123, 255, 0.40) !important; color: white !important;",
                                    },
                                ],
                            ),
                        ]
                    ),
                ],
                className="mb-3",
            ),
            # Extra metrics (participation / normalization)
            dbc.Card(
                [
                    dbc.CardHeader("Baseline vs ML Extra Metrics (Participation / Normalization)"),
                    dbc.CardBody(
                        [
                            dash_table.DataTable(
                                id="mlcpo-metrics2-table",
                                columns=[],
                                data=[],
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "fontSize": "0.85rem",
                                    "padding": "6px",
                                    "whiteSpace": "nowrap",
                                    "backgroundColor": "rgba(0,0,0,0)",
                                    "color": "white",
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "rgba(255,255,255,0.08)",
                                    "color": "white",
                                },
                                css=[
                                    # Row hover -> blue highlight instead of white
                                    {
                                        "selector": ".dash-spreadsheet-container .dash-spreadsheet-inner tr:hover td",
                                        "rule": "background-color: rgba(0, 0, 255, 0.7) !important; color: white !important;",
                                    },
                                    # Also keep active/selected readable (click focus)
                                    {
                                        "selector": ".dash-spreadsheet-container .dash-spreadsheet-inner td.focused, "
                                                    ".dash-spreadsheet-container .dash-spreadsheet-inner td.cell--selected",
                                        "rule": "background-color: rgba(0, 123, 255, 0.40) !important; color: white !important;",
                                    },
                                ],
                            ),
                        ]
                    ),
                ],
                className="mb-3",
            ),



            # Lower: cycle summaries table
            dbc.Card(
                [
                    dbc.CardHeader("Cycle Summary"),
                    dbc.CardBody(
                        [
                            dash_table.DataTable(
                                id="mlcpo-cycle-table",
                                columns=[],
                                data=[],
                                page_size=15,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "fontSize": "0.80rem",
                                    "padding": "6px",
                                    "whiteSpace": "nowrap",
                                    "backgroundColor": "rgba(0,0,0,0)",
                                    "color": "white",
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "rgba(255,255,255,0.08)",
                                    "color": "white",
                                },
                                css=[
                                    # Row hover -> blue highlight instead of white
                                    {
                                        "selector": ".dash-spreadsheet-container .dash-spreadsheet-inner tr:hover td",
                                        "rule": "background-color: rgba(0, 0, 255, 0.7) !important; color: white !important;",
                                    },
                                    # Also keep active/selected readable (click focus)
                                    {
                                        "selector": ".dash-spreadsheet-container .dash-spreadsheet-inner td.focused, "
                                                    ".dash-spreadsheet-container .dash-spreadsheet-inner td.cell--selected",
                                        "rule": "background-color: rgba(0, 123, 255, 0.40) !important; color: white !important;",
                                    },
                                ],
                            ),
                        ]
                    ),
                    dbc.Modal(
                        [
                            dbc.ModalHeader(
                                [
                                    dbc.ModalTitle("ML CPO — Equity & Drawdown (Baseline vs ML)"),
                                    dbc.Button(
                                        "Close",
                                        id="mlcpo-close-charts-btn",
                                        color="secondary",
                                        outline=True,
                                        size="sm",
                                        className="ms-2",
                                    ),
                                ],
                                className="d-flex justify-content-between align-items-center",
                            ),
                            dbc.ModalBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Equity view", style={"marginBottom": "2px"}),
                                                    dbc.RadioItems(
                                                        id="mlcpo-equity-mode",
                                                        options=[
                                                            {"label": "Raw P&L (cumulative)", "value": "raw"},
                                                            {"label": "P&L per UID/day (cumulative)", "value": "per_uid"},
                                                            {"label": "P&L per Margin/day (cumulative)", "value": "per_margin"},
                                                            {"label": "PCR % (daily)", "value": "pcr_daily_pct"},
                                                            {"label": "PCR % (rolling 20d)", "value": "pcr_roll20_pct"},
                                                            {"label": "PCR % (rolling 60d)", "value": "pcr_roll60_pct"},
                                                        ],
                                                        value="raw",
                                                        inline=True,
                                                        style={"fontSize": "0.90rem"},
                                                    ),

                                                ],
                                                width=8,
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Drawdown view", style={"marginBottom": "2px"}),
                                                    dbc.RadioItems(
                                                        id="mlcpo-dd-mode",
                                                        options=[
                                                            {"label": "DD ($)", "value": "dd_$"},
                                                            {"label": "DD (%)", "value": "dd_pct"},
                                                        ],
                                                        value="dd_$",
                                                        inline=True,
                                                        style={"fontSize": "0.90rem"},
                                                    ),
                                                ],
                                                width=4,
                                            ),
                                        ],
                                        className="g-2",
                                        style={"marginBottom": "8px"},
                                    ),
                                    dcc.Graph(id="mlcpo-equity-graph", figure={}, config={"displayModeBar": True}),
                                    dcc.Graph(id="mlcpo-dd-graph", figure={}, config={"displayModeBar": True}),
                                ]
                            ),

                        ],
                        id="mlcpo-charts-modal",
                        is_open=False,
                        size="xl",
                        fullscreen=True,
                        scrollable=True,
                    )

                ]
            ),
            dcc.Store(id="mlcpo-mode", data="monthly"),
            # Store last run results (meta only)
            dcc.Store(id="mlcpo-last-run-meta", data=None),
        ],
        fluid=True,
    )


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
@callback(
    Output("mlcpo-dataset-dropdown", "options"),
    Input("mlcpo-refresh-datasets-btn", "n_clicks"),
    prevent_initial_call=False,
)
def mlcpo_refresh_datasets(_n):
    idx = _load_ml_datasets_index()
    return _dataset_options_from_index(idx)


@callback(
    Output("mlcpo-mode", "data"),
    Output("mlcpo-monthly-params", "style"),
    Output("mlcpo-weekly-params", "style"),
    Output("mlcpo-mode-hint", "children"),
    Output("mlcpo-run-btn", "children"),
    Output("mlcpo-mode-monthly-btn", "color"),
    Output("mlcpo-mode-weekly-btn", "color"),
    Input("mlcpo-mode-monthly-btn", "n_clicks"),
    Input("mlcpo-mode-weekly-btn", "n_clicks"),
    State("mlcpo-mode", "data"),
    prevent_initial_call=False,
)
def mlcpo_set_mode(n_monthly, n_weekly, current_mode):
    trig = ctx.triggered_id

    # Initial page load: keep current_mode (defaults to "monthly")
    mode = current_mode or "monthly"
    if trig == "mlcpo-mode-monthly-btn":
        mode = "monthly"
    elif trig == "mlcpo-mode-weekly-btn":
        mode = "weekly"

    if mode == "weekly":
        monthly_style = {"display": "none"}
        weekly_style = {"display": "block"}
        hint = "Mode: Weekly (CPO)"
        run_label = "Run ML (Weekly CPO)"
        monthly_color = "secondary"
        weekly_color = "primary"
    else:
        monthly_style = {"display": "block"}
        weekly_style = {"display": "none"}
        hint = "Mode: Monthly (Static)"
        run_label = "Run ML (Monthly Static)"
        monthly_color = "primary"
        weekly_color = "secondary"

    return mode, monthly_style, weekly_style, hint, run_label, monthly_color, weekly_color


def _to_float_list_safe(s):
    out = []
    for v in s:
        if v is None:
            out.append(0.0)
            continue
        try:
            fv = float(v)
            # NaN check: NaN != NaN
            if fv != fv:
                out.append(0.0)
            else:
                out.append(fv)
        except Exception:
            out.append(0.0)
    return out


@callback(
    Output("mlcpo-selection-param-label", "children"),
    Input("mlcpo-selection-mode", "value"),
)
def mlcpo_update_selection_param_label(selection_mode):
    """
    Keep the numeric input label in sync with the selection mode:
    - Top-K / Bottom-K: interpret value as K (count of strategies)
    - Bottom-p: interpret value as p (percentile, e.g. 25 -> bottom 25%)
    """
    if selection_mode == "bottomk_per_day":
        return "Bottom K"
    elif selection_mode == "bottomp_perc":
        return "Bottom p (%)"
    else:
        # default / topk
        return "Top K"



@callback(
    Output("mlcpo-status", "children"),
    Output("mlcpo-status", "color"),
    Output("mlcpo-summary-text", "children"),
    Output("mlcpo-metrics-table", "columns"),
    Output("mlcpo-metrics-table", "data"),
    Output("mlcpo-metrics2-table", "columns"),
    Output("mlcpo-metrics2-table", "data"),
    Output("mlcpo-cycle-table", "columns"),
    Output("mlcpo-cycle-table", "data"),
    Output("mlcpo-curves-store", "data"),
    Output("mlcpo-last-run-meta", "data"),
    Input("mlcpo-run-btn", "n_clicks"),
    State("mlcpo-dataset-dropdown", "value"),
    State("mlcpo-start-date", "value"),
    State("mlcpo-end-date", "value"),
    State("mlcpo-is-months", "value"),
    State("mlcpo-oos-months", "value"),
    State("mlcpo-step-months", "value"),
    State("mlcpo-anchored-type", "value"),
    State("mlcpo-selection-mode", "value"),
    State("mlcpo-top-k", "value"),
    State("mlcpo-mode", "data"),
    State("mlcpo-oos-weeks", "value"),
    State("mlcpo-step-weeks", "value"),
    running=[(Output("mlcpo-run-btn", "disabled"), True, False)],
    prevent_initial_call=True,
)
def mlcpo_run_fwa(
    n_clicks,
    dataset_path,
    start_date,
    end_date,
    is_months,
    oos_months,
    step_months,
    anchored_type,
    selection_mode,
    top_k,
    mode,
    oos_weeks,
    step_weeks,
):
    
    print("ML CPO dataset_path =", repr(dataset_path))
    
    if dataset_path is None or dataset_path == "":
        return (
            "ERROR: No ML dataset selected. Choose one from the dropdown.",
            "danger",
            "",
            [],
            [],
            [],
            [],
            [],
            [],
            None,
            None,
        )



    # Basic validation (keep strict; no silent coercion)
    start_date = (start_date or "").strip() or None
    end_date = (end_date or "").strip() or None

    # --- NEW: allow Top-K and Bottom-K selection modes from the UI ---
    if selection_mode not in ("topk_per_day", "bottomk_per_day", "bottomp_perc"):
        return (
            "ERROR: Unsupported selection mode.",
            "danger",
            "",
            [],
            [],
            [],
            [],
            [],
            [],
            None,
            None,
        )
    
    curves_payload = None # so it avoids error when run in MOnthly ml1.py that has NO charts in it
    
    try:
        mode = (mode or "monthly").strip().lower()

        if mode == "weekly":
            # Weekly CPO (ml2): IS in months, OoS/Step in weeks

            # --- NEW: map UI selection_mode -> internal flag for ml2.RunParamsWeekly ---
            if selection_mode == "topk_per_day":
                selection_mode_internal = "top_k"
            elif selection_mode == "bottomk_per_day":
                selection_mode_internal = "bottom_k"
            elif selection_mode == "bottomp_perc":
                selection_mode_internal = "bottom_p"
            else:
                # Should not happen because of the earlier validation, but keep a hard guard
                return (
                    "ERROR: Unsupported selection mode.",
                    "danger",
                    "",
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    None,
                    None,
                )

            params2 = RunParamsWeekly(
                dataset_csv_path=dataset_path,
                start_date=start_date,
                end_date=end_date,
                is_months=int(is_months),
                oos_weeks=int(oos_weeks),
                step_weeks=int(step_weeks),
                anchored_type=str(anchored_type or "U"),
                top_k_per_day=int(top_k),
                selection_mode=selection_mode_internal,  # <<< NEW: pass through to ml2.py
                verbose_cycles=False,  # UI should not spam console
            )
            out = run_fwa_weekly(params2)

            
            bcurve = out.get("baseline_curve")
            mcurve = out.get("ml_curve")
            
            curves_payload = None
            try:
                if bcurve is not None and not bcurve.empty and mcurve is not None and not mcurve.empty:
                    curves_payload = {
                        "baseline": {
                            "date": bcurve["date"].astype(str).tolist(),
                            "pnl_day": bcurve["pnl_day"].astype(float).tolist(),
                            "uid_day": bcurve["uid_day"].astype(float).tolist(),
                            "margin_day": bcurve["margin_day"].astype(float).tolist(),
                            "dd_raw": bcurve["dd_raw"].astype(float).tolist(),
                            "dd_raw_pct": bcurve["dd_raw_pct"].astype(float).tolist(),
                            "premium_day": _to_float_list_safe(bcurve["premium_day"].tolist()),

                        },
                        "ml": {
                            "date": mcurve["date"].astype(str).tolist(),
                            "pnl_day": mcurve["pnl_day"].astype(float).tolist(),
                            "uid_day": mcurve["uid_day"].astype(float).tolist(),
                            "margin_day": mcurve["margin_day"].astype(float).tolist(),
                            "dd_raw": mcurve["dd_raw"].astype(float).tolist(),
                            "dd_raw_pct": mcurve["dd_raw_pct"].astype(float).tolist(),
                            "premium_day": _to_float_list_safe(mcurve["premium_day"].tolist()),
                        },
                    }
            except Exception:
                curves_payload = None


        else:
            # Monthly Static (ml1): IS/OoS/Step in months
            params1 = RunParams(
                dataset_csv_path=dataset_path,
                start_date=start_date,
                end_date=end_date,
                is_months=int(is_months),
                oos_months=int(oos_months),
                anchored_type=str(anchored_type or "U"),
                step_months=int(step_months),
                top_k_per_day=int(top_k),
                verbose_cycles=False,  # UI should not spam console
            )
            out = run_fwa_single(params1)

        bm = out.get("baseline_metrics", {}) or {}
        mm = out.get("ml_metrics", {}) or {}

        def _fmt_pct(x):
            try:
                return f"{float(x)*100:.2f}%"
            except Exception:
                return ""

        def _fmt_money(x):
            try:
                return f"{float(x):,.2f}"
            except Exception:
                return ""

        def _fmt_num(x):
            try:
                return f"{float(x):.4f}"
            except Exception:
                return ""

        # Table rows (you asked: ALL metrics for both)
        metrics_rows = [
            {
                "set": "BASELINE",
                "trades": bm.get("trades"),
                "total_pnl_$": _fmt_money(bm.get("total_pnl")),
                "total_pnlR": _fmt_num(bm.get("total_pnlR")),
                "return_%": _fmt_pct(bm.get("return_pct")),
                "PCR": _fmt_pct(bm.get("pcr")),
                "max_DD_$": _fmt_money(bm.get("max_dd_$")),
                "max_DD_%": _fmt_pct(bm.get("max_dd_%")),
                "sharpe_daily": _fmt_num(bm.get("sharpe_daily")),
                "win_month_%": _fmt_pct(bm.get("win_month_pct")),
                "avg_month_pnl_$": _fmt_money(bm.get("avg_month_pnl")),
                "median_month_pnl_$": _fmt_money(bm.get("median_month_pnl")),
                "best_month_pnl_$": _fmt_money(bm.get("best_month_pnl")),
                "worst_month_pnl_$": _fmt_money(bm.get("worst_month_pnl")),
            },
            {
                "set": "ML",
                "trades": mm.get("trades"),
                "total_pnl_$": _fmt_money(mm.get("total_pnl")),
                "total_pnlR": _fmt_num(mm.get("total_pnlR")),
                "return_%": _fmt_pct(mm.get("return_pct")),
                "PCR": _fmt_pct(mm.get("pcr")),
                "max_DD_$": _fmt_money(mm.get("max_dd_$")),
                "max_DD_%": _fmt_pct(mm.get("max_dd_%")),
                "sharpe_daily": _fmt_num(mm.get("sharpe_daily")),
                "win_month_%": _fmt_pct(mm.get("win_month_pct")),
                "avg_month_pnl_$": _fmt_money(mm.get("avg_month_pnl")),
                "median_month_pnl_$": _fmt_money(mm.get("median_month_pnl")),
                "best_month_pnl_$": _fmt_money(mm.get("best_month_pnl")),
                "worst_month_pnl_$": _fmt_money(mm.get("worst_month_pnl")),
            },
        ]

        metrics_cols = [{"name": k, "id": k} for k in metrics_rows[0].keys()]
        
        bx = out.get("baseline_extra_metrics", {}) or {}
        mx = out.get("ml_extra_metrics", {}) or {}
        
        def _fmt_int(x):
            try:
                return str(int(x))
            except Exception:
                return ""
        
        extra_rows = [
            {
                "set": "BASELINE",
                "nominal_units": _fmt_int(bx.get("nominal_units")),
                "avg_trades_day": _fmt_num(bx.get("avg_trades_day")),
                "med_trades_day": _fmt_num(bx.get("med_trades_day")),
                "p95_trades_day": _fmt_num(bx.get("p95_trades_day")),
                "max_trades_day": _fmt_num(bx.get("max_trades_day")),
                "avg_unique_uid_day": _fmt_num(bx.get("avg_unique_uid_day")),
                "med_unique_uid_day": _fmt_num(bx.get("med_unique_uid_day")),
                "p95_unique_uid_day": _fmt_num(bx.get("p95_unique_uid_day")),
                "max_unique_uid_day": _fmt_num(bx.get("max_unique_uid_day")),
                "p95_margin_day": _fmt_money(bx.get("p95_margin_day")),
                "max_margin_day": _fmt_money(bx.get("max_margin_day")),
                "p95_abs_premium_day": _fmt_money(bx.get("p95_abs_premium_day")),
                "max_abs_premium_day": _fmt_money(bx.get("max_abs_premium_day")),
                "total_pnl_per_nominal_unit": _fmt_money(bx.get("total_pnl_per_nominal_unit")),
                "total_pnlR_per_nominal_unit": _fmt_num(bx.get("total_pnlR_per_nominal_unit")),
                "total_pnl_per_avg_active_uid": _fmt_money(bx.get("total_pnl_per_avg_active_uid")),
                "total_pnlR_per_avg_active_uid": _fmt_num(bx.get("total_pnlR_per_avg_active_uid")),
            },
            {
                "set": "ML",
                "nominal_units": _fmt_int(mx.get("nominal_units")),
                "avg_trades_day": _fmt_num(mx.get("avg_trades_day")),
                "med_trades_day": _fmt_num(mx.get("med_trades_day")),
                "p95_trades_day": _fmt_num(mx.get("p95_trades_day")),
                "max_trades_day": _fmt_num(mx.get("max_trades_day")),
                "avg_unique_uid_day": _fmt_num(mx.get("avg_unique_uid_day")),
                "med_unique_uid_day": _fmt_num(mx.get("med_unique_uid_day")),
                "p95_unique_uid_day": _fmt_num(mx.get("p95_unique_uid_day")),
                "max_unique_uid_day": _fmt_num(mx.get("max_unique_uid_day")),
                "p95_margin_day": _fmt_money(mx.get("p95_margin_day")),
                "max_margin_day": _fmt_money(mx.get("max_margin_day")),
                "p95_abs_premium_day": _fmt_money(mx.get("p95_abs_premium_day")),
                "max_abs_premium_day": _fmt_money(mx.get("max_abs_premium_day")),
                "total_pnl_per_nominal_unit": _fmt_money(mx.get("total_pnl_per_nominal_unit")),
                "total_pnlR_per_nominal_unit": _fmt_num(mx.get("total_pnlR_per_nominal_unit")),
                "total_pnl_per_avg_active_uid": _fmt_money(mx.get("total_pnl_per_avg_active_uid")),
                "total_pnlR_per_avg_active_uid": _fmt_num(mx.get("total_pnlR_per_avg_active_uid")),
            },
        ]
        
        extra_cols = [{"name": k, "id": k} for k in extra_rows[0].keys()]



        n_cycles = len(out.get("cycles", []))

        summary = (
            f"Dataset: {dataset_path}\n"
            f"Cycles: {n_cycles}\n"
            f"Initial Equity: {100000:.0f}\n"
            f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )


        cy = out.get("cycle_summaries")
        if isinstance(cy, pd.DataFrame) and not cy.empty:
            cols = [{"name": c, "id": c} for c in cy.columns.tolist()]
            data = cy.to_dict("records")
        else:
            cols, data = [], []

        meta = {
            "dataset_path": dataset_path,
            "baseline_metrics": bm,
            "ml_metrics": mm,
            "n_cycles": n_cycles,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }


        return (
            "OK: ML FWA run completed.",
            "success",
            summary,
            metrics_cols,
            metrics_rows,
            extra_cols,
            extra_rows,
            cols,
            data,
            curves_payload,
            meta,
        )



    except Exception as e:
        return (
            f"ERROR: ML run failed: {e}",
            "danger",
            "",
            [],
            [],
            [],
            [],
            [],
            [],
            None,
            None,
        )


def _cum_sum(vals):
    out = []
    s = 0.0
    for v in vals:
        s += float(v)
        out.append(s)
    return out

def _make_equity_fig(curves_payload: dict, mode: str):
    """
    mode:
      - raw: cum(pnl_day)
      - per_uid: cum(pnl_day / uid_day)
      - per_margin: cum(pnl_day / margin_day)
    """
    fig = go.Figure()
    
    def _rolling_sum(vals, window: int):
        out = []
        s = 0.0
        q = []
        for v in vals:
            v = float(v)
            q.append(v)
            s += v
            if len(q) > window:
                s -= q.pop(0)
            out.append(s)
        return out


    def _series(label: str, series: dict):
        dates = series.get("date", []) or []
        pnl = series.get("pnl_day", []) or []
        uid = series.get("uid_day", []) or []
        mgn = series.get("margin_day", []) or []
        prem = series.get("premium_day", []) or []

        n = min(len(dates), len(pnl), len(uid), len(mgn), len(prem))

        if n == 0:
            return [], []

        dates = dates[:n]
        def _f(x):
            try:
                if x is None:
                    return 0.0
                v = float(x)
                return 0.0 if v != v else v  # NaN -> 0
            except Exception:
                return 0.0
        
        pnl = [_f(x) for x in pnl[:n]]
        uid = [_f(x) for x in uid[:n]]
        mgn = [_f(x) for x in mgn[:n]]
        prem = [_f(x) for x in prem[:n]]


        if mode == "raw":
            inc = pnl
            ytitle = "Cumulative P&L ($)"

        elif mode == "per_uid":
            # normalize daily pnl by unique strategy count that day
            inc = []
            for p, u in zip(pnl, uid):
                inc.append(p / u if u and u > 0 else 0.0)
            ytitle = "Cumulative P&L per UID ($)"

        elif mode == "per_margin":
            # normalize daily pnl by margin used that day (unitless). Multiply by 10,000 for readability if you want.
            inc = []
            for p, mg in zip(pnl, mgn):
                inc.append(p / mg if mg and mg > 0 else 0.0)
            ytitle = "Cumulative P&L per $1 Margin (ratio)"

        elif mode == "pcr_daily_pct":
            # PCR % per day = 100 * pnl_day / abs(premium_day)
            y = []
            for p, pr in zip(pnl, prem):
                denom = abs(pr)
                y.append(100.0 * (p / denom) if denom > 0 else 0.0)
            ytitle = "PCR (%) — daily"
            return dates, y, ytitle

        elif mode == "pcr_roll20_pct":
            # PCR % rolling 20d = 100 * sum(pnl) / sum(abs(premium))
            prem_abs = [abs(x) for x in prem]
            pnl_sum = _rolling_sum(pnl, 20)
            prem_sum = _rolling_sum(prem_abs, 20)
            y = []
            for ps, prs in zip(pnl_sum, prem_sum):
                y.append(100.0 * (ps / prs) if prs > 0 else 0.0)
            ytitle = "PCR (%) — rolling 20d"
            return dates, y, ytitle
        elif mode == "pcr_roll60_pct":
            # PCR % rolling 60d = 100 * sum(pnl) / sum(abs(premium))
            prem_abs = [abs(x) for x in prem]
            pnl_sum = _rolling_sum(pnl, 60)
            prem_sum = _rolling_sum(prem_abs, 60)
            y = []
            for ps, prs in zip(pnl_sum, prem_sum):
                y.append(100.0 * (ps / prs) if prs > 0 else 0.0)
            ytitle = "PCR (%) — rolling 60d"
            return dates, y, ytitle


        else:
            inc = pnl
            ytitle = "Cumulative P&L ($)"

        y = _cum_sum(inc)
        return dates, y, ytitle

    b = curves_payload.get("baseline") or {}
    m = curves_payload.get("ml") or {}

    xb, yb, ytitle = _series("BASELINE", b)
    xm, ym, _ = _series("ML", m)

    fig.add_trace(go.Scatter(x=xb, y=yb, mode="lines", name="BASELINE"))
    fig.add_trace(go.Scatter(x=xm, y=ym, mode="lines", name="ML"))

    fig.update_layout(
        title="Equity (cumulative) — Raw vs Exposure-normalized",
        yaxis_title=ytitle,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_dark",
        legend=dict(orientation="h"),
    )

    return fig


def _make_dd_fig(curves_payload: dict, mode: str):
    fig = go.Figure()
    b = curves_payload.get("baseline") or {}
    m = curves_payload.get("ml") or {}

    def _dd(series):
        dates = series.get("date", []) or []
        if mode == "dd_pct":
            ddp = series.get("dd_raw_pct", []) or []
            y = [100.0 * float(x) for x in ddp[:len(dates)]]
            return dates, y, "Drawdown (%)"
        dd = series.get("dd_raw", []) or []
        y = [float(x) for x in dd[:len(dates)]]
        return dates, y, "Drawdown ($)"

    xb, yb, ytitle = _dd(b)
    xm, ym, _ = _dd(m)

    fig.add_trace(go.Scatter(x=xb, y=yb, mode="lines", name="BASELINE"))
    fig.add_trace(go.Scatter(x=xm, y=ym, mode="lines", name="ML"))

    fig.update_layout(
        title="Drawdown (realized, raw)",
        yaxis_title=ytitle,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_dark",
        legend=dict(orientation="h"),
    )
    return fig




@callback(
    Output("mlcpo-charts-modal", "is_open"),
    Output("mlcpo-equity-graph", "figure"),
    Output("mlcpo-dd-graph", "figure"),
    Input("mlcpo-open-charts-btn", "n_clicks"),
    Input("mlcpo-close-charts-btn", "n_clicks"),
    Input("mlcpo-equity-mode", "value"),
    Input("mlcpo-dd-mode", "value"),
    State("mlcpo-charts-modal", "is_open"),
    State("mlcpo-curves-store", "data"),
    prevent_initial_call=True,
)

def mlcpo_toggle_charts(open_n, close_n, equity_mode, dd_mode, is_open, curves_payload):
    trig = ctx.triggered_id

    # Close
    if trig == "mlcpo-close-charts-btn":
        return False, {}, {}

    # If no data, do not crash; open modal but show empty figs
    if not curves_payload or not curves_payload.get("baseline") or not curves_payload.get("ml"):
        if trig == "mlcpo-open-charts-btn":
            return False, {}, {}
        return is_open, {}, {}

    # Open (or update while open when modes change)
    if trig == "mlcpo-open-charts-btn":
        eq = _make_equity_fig(curves_payload, equity_mode or "raw")
        dd = _make_dd_fig(curves_payload, dd_mode or "dd_$")

        return True, eq, dd

    # Mode toggles should update figures only if modal is open
    if trig in ("mlcpo-equity-mode", "mlcpo-dd-mode"):
        if not is_open:
            return is_open, {}, {}
        eq = _make_equity_fig(curves_payload, equity_mode or "raw")
        dd = _make_dd_fig(curves_payload, dd_mode or "dd_$")
        return is_open, eq, dd

    return is_open, {}, {}


