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
                                                                {"label": "Top-K per day (by predicted prob)", "value": "topk_per_day"},
                                                            ],
                                                            value="topk_per_day",
                                                            className="p26-dark-dropdown",
                                                            clearable=False,
                                                            style={"fontSize": "0.85rem"},
                                                        ),
                                                    ],
                                                    width=8,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Top K"),
                                                        dbc.Input(
                                                            id="mlcpo-top-k",
                                                            type="number",
                                                            value=3,
                                                            min=1,
                                                            step=1,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    width=2,
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
                            ),
                        ]
                    ),
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




@callback(
    Output("mlcpo-status", "children"),
    Output("mlcpo-status", "color"),
    Output("mlcpo-summary-text", "children"),
    Output("mlcpo-metrics-table", "columns"),
    Output("mlcpo-metrics-table", "data"),
    Output("mlcpo-cycle-table", "columns"),
    Output("mlcpo-cycle-table", "data"),
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
            None,
        )



    # Basic validation (keep strict; no silent coercion)
    start_date = (start_date or "").strip() or None
    end_date = (end_date or "").strip() or None

    if selection_mode != "topk_per_day":
        return (
            "ERROR: Unsupported selection mode.",
            "danger",
            "",
            [],
            [],
            [],
            [],
            None,
        )

    try:
        mode = (mode or "monthly").strip().lower()

        if mode == "weekly":
            # Weekly CPO (ml2): IS in months, OoS/Step in weeks
            params2 = RunParamsWeekly(
                dataset_csv_path=dataset_path,
                start_date=start_date,
                end_date=end_date,
                is_months=int(is_months),
                oos_weeks=int(oos_weeks),
                step_weeks=int(step_weeks),
                anchored_type=str(anchored_type or "U"),
                top_k_per_day=int(top_k),
                verbose_cycles=False,  # UI should not spam console
            )
            out = run_fwa_weekly(params2)

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
            cols,
            data,
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
            None,
        )

