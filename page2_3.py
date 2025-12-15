
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 15:56:01 2025

@author: mauro

Phase 2 – Portfolio Builder (v1)

Goal:
- Build/adjust a discrete integer-lot portfolio for the ACTIVE+SELECTED strategies,
  under risk/budget constraints, with a preview-only workflow:
  - Run -> show baseline vs candidate metrics + proposed lots
  - Apply -> write candidate lots into p2-weights-store (as factors=lots)
  - Discard -> no state change

Notes:p2-weights-store
- Uses page2._build_portfolio_timeseries for portfolio daily P&L series.
- Does NOT save anything to registry (preview only).
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, ctx, no_update
import numpy as np
import pandas as pd

#from pages.page2 import _build_portfolio_timeseries  # reuse existing engine


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------

def layout_tab_portfolio_builder():
    return html.Div(
        [
            # ---- Controls
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "Run mode:",
                                            style={"marginRight": "0.5rem", "fontSize": "0.85rem"},
                                        ),
                                        dcc.Dropdown(
                                            id="p2-builder-run-mode",
                                            options=[
                                                {"label": "Fast (repair only)", "value": "fast"},
                                                {"label": "Balanced (local search)", "value": "balanced"},
                                                {"label": "Thorough (more iterations)", "value": "thorough"},
                                            ],
                                            value="balanced",
                                            clearable=False,
                                            style={"fontSize": "0.85rem", "width": "200px", "color": "blue"},
                                        ),
                                    ],
                                    style={"display": "flex", "alignItems": "center", "gap": "0.5rem"},
                                ),
                            ]
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Span(
                                    "Initial equity:",
                                    style={"marginRight": "0.5rem", "fontSize": "0.85rem"},
                                ),
                                dcc.Input(
                                    id="p2-builder-initial-equity-input",
                                    type="number",
                                    value=100000,
                                    min=0,
                                    step=10000,
                                    style={
                                        "width": "140px",
                                        "fontSize": "0.85rem",
                                        "backgroundColor": "#2a2a2a",
                                        "color": "#EEEEEE",
                                        "border": "1px solid #555555",
                                    },
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"},
                        ),
                        md=6,
                    ),
                ],
                className="mb-2",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Constraints (v1)"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(_num_input("Max total lots", "p2-builder-max-total-lots", 25, 0, 1), md=3),
                                                dbc.Col(_num_input("Min lots / strat", "p2-builder-min-lots", 0, 0, 1), md=3),
                                                dbc.Col(_num_input("Max lots / strat", "p2-builder-max-lots", 3, 0, 1), md=3),
                                                dbc.Col(_num_input("Budget (max margin)", "p2-builder-max-budget", None, 0, 1000), md=3),
                                            ],
                                            className="mb-2",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(_num_input("Max daily loss ($)", "p2-builder-max-daily-loss", None, 0, 100), md=3),
                                                dbc.Col(_num_input("Max DD ($)", "p2-builder-max-dd", None, 0, 100), md=3),
                                                dbc.Col(_num_input("Max losing streak (days)", "p2-builder-max-losing-streak", None, 0, 1), md=3),
                                                dbc.Col(_num_input("Max DD duration (days)", "p2-builder-max-dd-duration", None, 0, 1), md=3),
                                            ],
                                        ),
                                        html.Div(
                                            "Note: v1 focuses on risk constraints. Performance is used only as a tie-breaker.",
                                            style={"fontSize": "0.75rem", "color": "#AAAAAA", "marginTop": "0.5rem"},
                                        ),
                                    ]
                                ),
                            ],
                            style={"backgroundColor": "#222222", "border": "1px solid #444444"},
                        ),
                        md=12,
                    ),
                ],
                className="mb-2",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                dbc.Button("Run Builder", id="p2-builder-run-btn", color="primary", size="sm", className="me-2"),
                                dbc.Button("Apply", id="p2-builder-apply-btn", color="success", size="sm", className="me-2"),
                                dbc.Button("Discard", id="p2-builder-discard-btn", color="secondary", outline=True, size="sm"),
                                dcc.Loading(
                                    id="p2-builder-status-loading",
                                    type="default",
                                    color="#BBBBBB",
                                    children=html.Span(
                                        id="p2-builder-status-local",
                                        style={
                                            "marginLeft": "1rem",
                                            "fontSize": "0.85rem",
                                            "color": "#DDDDDD",
                                            "flex": "1 1 0",
                                            "minWidth": 0,
                                        },
                                    ),
                                ),

                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                        md=12,
                    ),
                ],
                className="mb-3",
            ),

            # ---- Preview outputs
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div("Baseline vs Candidate (key metrics)", style={"fontSize": "0.85rem", "fontWeight": "bold"}),
                                html.Div(id="p2-builder-metrics-compare", style={"marginTop": "0.5rem", "fontSize": "0.85rem"}),
                            ],
                            style={"backgroundColor": "#222222", "border": "1px solid #444444", "padding": "0.75rem"},
                        ),
                        md=5,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Div("Proposed integer lots (candidate)", style={"fontSize": "0.85rem", "fontWeight": "bold"}),
                                html.Div(id="p2-builder-lots-table", style={"marginTop": "0.5rem", "fontSize": "0.8rem", "maxHeight": "420px", "overflowY": "auto"}),
                            ],
                            style={"backgroundColor": "#222222", "border": "1px solid #444444", "padding": "0.75rem"},
                        ),
                        md=7,
                    ),
                ]
            ),

            dcc.Store(id="p2-builder-last-run", data=None),  # holds candidate lots + metrics for Apply
        ],
        style={"padding": "0.75rem", "fontSize": "0.85rem"},
    )


def _num_input(label, component_id, default, min_val, step):
    return html.Div(
        [
            html.Div(label, style={"fontSize": "0.75rem", "color": "#AAAAAA", "marginBottom": "0.2rem"}),
            dcc.Input(
                id=component_id,
                type="number",
                value=default,
                min=min_val,
                step=step,
                style={
                    "width": "100%",
                    "fontSize": "0.85rem",
                    "backgroundColor": "#2a2a2a",
                    "color": "#EEEEEE",
                    "border": "1px solid #555555",
                },
            ),
        ]
    )


# -----------------------------------------------------------------------------
# Core helpers (v1)
# -----------------------------------------------------------------------------

def _selected_active_rows(active_store):
    """
    Must match page2.py selection logic:
    use only rows with is_selected == True.
    """
    active_store = active_store or []
    return [r for r in active_store if r.get("is_selected")]



def _get_uid_name_map(active_store):
    rows = _selected_active_rows(active_store)
    uid_to_name = {}
    for r in rows:
        uid = r.get("uid") or r.get("id") or r.get("file_path")
        name = r.get("name", uid)
        if uid is not None:
            uid_to_name[uid] = name
    return uid_to_name


def _lots_from_weights(weights_store, uids, default_lot=1):
    weights_store = weights_store or {}
    lots = {}
    for uid in uids:
        w = weights_store.get(uid, {})
        factor = w.get("factor", None)
        if factor is None:
            lots[uid] = int(default_lot)
        else:
            try:
                lots[uid] = int(np.clip(int(round(float(factor))), 0, 10**9))
            except Exception:
                lots[uid] = int(default_lot)
    return lots


def _weights_from_lots(existing_weights_store, lots_map):
    """
    Convert integer lots to weights_store format (factor as float).
    Keep any unrelated keys in store.
    """
    existing_weights_store = existing_weights_store or {}
    out = dict(existing_weights_store)
    for uid, lot in lots_map.items():
        out[uid] = {"factor": float(int(lot))}
    return out


def _portfolio_series(active_store, weights_store, initial_equity, weight_mode="lots"):
    """
    Wrapper for page2._build_portfolio_timeseries.

    Builder always works in integer-lot space:
    - weights_store encodes lots via factor = lot
    - weight_mode="lots" tells the engine to interpret factors as lots and
      scale P&L accordingly.
    """
    from pages.page2 import _build_portfolio_timeseries  # lazy import to avoid circular import

    initial_equity = float(initial_equity) if initial_equity is not None else 100000.0

    return _build_portfolio_timeseries(
        active_store=active_store or [],
        weights_store=weights_store or {},
        weight_mode=weight_mode or "lots",
        initial_equity=initial_equity,
    )



def _compute_key_metrics(ts_result):
    """
    Extract key risk metrics from the series dict returned by
    _build_portfolio_timeseries.
    """
    if not ts_result:
        return None

    daily = ts_result.get("portfolio_daily")
    # fallback if we ever change naming in the future
    if daily is None:
        daily = ts_result.get("portfolio_daily_pnl")

    dd = ts_result.get("dd")

    if daily is None:
        return None

    daily = pd.Series(daily).dropna()
    total_pnl = float(daily.sum())
    max_daily_loss = float(-daily.min()) if float(daily.min()) < 0 else 0.0

    # losing streak on daily pnl < 0
    neg = (daily < 0).astype(int)
    streak = 0
    max_streak = 0
    for v in neg.values:
        if v == 1:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    max_dd_abs = None
    dd_duration = None
    if dd is not None:
        dd = pd.Series(dd).dropna()
        max_dd_abs = float(dd.max()) if len(dd) else 0.0  # dd is positive distance from equity peak

        in_dd = (dd > 0).astype(int)
        cur = 0
        best = 0
        for v in in_dd.values:
            if v == 1:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        dd_duration = int(best)

    return {
        "total_pnl": total_pnl,
        "max_daily_loss": max_daily_loss,
        "max_losing_streak": int(max_streak),
        "max_dd_abs": max_dd_abs,
        "max_dd_duration": dd_duration,
        "n_days": int(len(daily)),
    }



def _violations(metrics, constraints):
    """
    Returns list of violated constraint labels.
    """
    v = []
    if metrics is None:
        return ["no_data"]

    # constraints: allow None = ignore
    mdl = constraints.get("max_daily_loss")
    if mdl is not None and metrics["max_daily_loss"] is not None and metrics["max_daily_loss"] > mdl:
        v.append("max_daily_loss")

    mdd = constraints.get("max_dd")
    if mdd is not None and metrics["max_dd_abs"] is not None and metrics["max_dd_abs"] > mdd:
        v.append("max_dd")

    mls = constraints.get("max_losing_streak")
    if mls is not None and metrics["max_losing_streak"] is not None and metrics["max_losing_streak"] > mls:
        v.append("max_losing_streak")

    mddd = constraints.get("max_dd_duration")
    if mddd is not None and metrics["max_dd_duration"] is not None and metrics["max_dd_duration"] > mddd:
        v.append("max_dd_duration")

    return v


def _objective(metrics):
    """
    Risk-first objective (lower is better).
    Performance is tie-breaker only (handled outside).
    """
    if metrics is None:
        return 1e18
    max_dd = metrics["max_dd_abs"] if metrics["max_dd_abs"] is not None else 0.0
    max_dl = metrics["max_daily_loss"] if metrics["max_daily_loss"] is not None else 0.0
    streak = float(metrics["max_losing_streak"] or 0)
    ddd = float(metrics["max_dd_duration"] or 0)
    # weighted sum; conservative
    return (1.0 * max_dd) + (1.0 * max_dl) + (250.0 * streak) + (100.0 * ddd)


def _rank_candidate(metrics, violations, lots_map):
    """
    Build a ranking tuple for a candidate portfolio.

    Lower tuple = better. Ordering:

      - All feasible portfolios (no violations) come before infeasible ones.
      - Within the feasible region:
          1) Higher total P&L is better.
          2) Lower Max DD is better.
          3) Lower Max daily loss is better.
          4) More active strategies (lots > 0) is better.
          5) More total lots is better (within caps).
      - Among infeasible portfolios:
          1) Fewer violated constraints is better.
          2) Lower composite risk score is better.
          3) Higher total P&L is better.
    """
    if metrics is None:
        # push completely invalid candidates to the bottom
        return (1, 999, 1e18, 0.0, 1e18, 1e18, 0, 0)

    total_pnl = float(metrics.get("total_pnl", 0.0) or 0.0)
    max_dd_abs = float(metrics.get("max_dd_abs", 0.0) or 0.0)
    max_daily_loss = float(metrics.get("max_daily_loss", 0.0) or 0.0)
    streak = float(metrics.get("max_losing_streak", 0.0) or 0.0)
    dd_dur = float(metrics.get("max_dd_duration", 0.0) or 0.0)

    lots_int = [max(0, int(v)) for v in (lots_map or {}).values()]
    n_active = sum(1 for v in lots_int if v > 0)
    total_lots = sum(lots_int)

    n_viol = len(violations or [])
    feasible_flag = 0 if n_viol == 0 else 1

    # simple composite for ordering infeasible candidates
    risk_score = max_dd_abs + max_daily_loss + 250.0 * streak + 100.0 * dd_dur

    if feasible_flag == 0:
        # Feasible region: P&L first, then risk, then diversification / capacity.
        return (
            0,
            -total_pnl,
            max_dd_abs,
            max_daily_loss,
            -n_active,
            -total_lots,
            0,
            0,
        )
    else:
        # Infeasible: fewer violations, lower risk_score, then P&L and others.
        return (
            1,
            n_viol,
            risk_score,
            -total_pnl,
            max_dd_abs,
            max_daily_loss,
            -n_active,
            -total_lots,
        )
    
    

def _search_integer_lots(active_store, base_lots, constraints, run_mode, initial_equity):
    """
    v2 search: local ±1 integer lot search with hard risk constraints.

    Behaviour:
      - Always runs, even if only structural caps (max_total_lots, min/max lot) are set.
      - Can move both down and up in integer lots.
      - Enforces structural caps inside the neighbourhood generator:
          * min_lot <= lot_i <= max_lot
          * sum(lot_i) <= max_total_lots
          * If baseline had exposure (>0 total lots), do not allow a fully flat portfolio.
      - Treats risk constraints (max_daily_loss, max_dd, max_losing_streak,
        max_dd_duration) as hard: feasible portfolios have zero violations.

    Algorithm (hill-climbing on integer grid):
      1) Start from baseline integer lots (rounded from factors, clipped to bounds).
      2) Evaluate baseline metrics + violations.
      3) At each iteration, consider all neighbours obtained by ±1 lot
         on a single strategy.
      4) Evaluate each neighbour and rank with _rank_candidate.
      5) Move to the best neighbour if its rank is strictly better.
      6) Stop when no neighbour improves the rank or max iterations reached.

    Returns:
        best_lots: dict[uid -> int]            (candidate lots)
        best_metrics: dict or None             (None if no feasible portfolio found)
        best_violations: list[str] or ["no_feasible"]
    """
    rows = _selected_active_rows(active_store)
    uids = [r.get("uid") or r.get("sid") or r.get("file_path") for r in rows]
    uids = [u for u in uids if u is not None]

    if not uids:
        return {}, None, ["no_strategies"]

    min_lot = int(constraints.get("min_lot", 0) or 0)
    max_lot = int(constraints.get("max_lot", 3) or 3)
    max_total_lots = int(constraints.get("max_total_lots", 25) or 25)

    def total_lots(x: dict) -> int:
        return int(sum(int(v) for v in x.values()))

    # ---- 1) Initial lots from baseline, clipped to [min_lot, max_lot]
    lots_current = {}
    for uid in uids:
        v0 = int(base_lots.get(uid, 1))
        if v0 < min_lot:
            v0 = min_lot
        if v0 > max_lot:
            v0 = max_lot
        lots_current[uid] = v0

    # Cap total lots down to max_total_lots (simple repair)
    while total_lots(lots_current) > max_total_lots:
        # Reduce from largest positions first
        uid_max = max(lots_current, key=lambda k: lots_current[k])
        if lots_current[uid_max] > min_lot:
            lots_current[uid_max] -= 1
        else:
            break

    baseline_total_lots = total_lots(lots_current)

    # Avoid fully flat portfolio if baseline had exposure and we have capacity
    if baseline_total_lots > 0 and total_lots(lots_current) == 0 and max_total_lots > 0:
        lots_current[uids[0]] = max(1, min_lot)
        lots_current[uids[0]] = int(np.clip(lots_current[uids[0]], min_lot, max_lot))

    def eval_lots(lots_map):
        w = _weights_from_lots({}, lots_map)
        ts = _portfolio_series(active_store, w, initial_equity, weight_mode="lots")
        metrics = _compute_key_metrics(ts)
        if metrics is None:
            return None, None, None
        viol = _violations(metrics, constraints)
        rank = _rank_candidate(metrics, viol, lots_map)
        return metrics, viol, rank

    # ---- 2) Evaluate baseline
    metrics0, viol0, rank0 = eval_lots(lots_current)
    if metrics0 is None:
        return base_lots, None, ["no_data"]

    best_lots = dict(lots_current)
    best_metrics = metrics0
    best_viol = viol0
    best_rank = rank0

    mode = (run_mode or "balanced").lower()
    if mode.startswith("fast"):
        max_iter = 20
    elif mode.startswith("thorough"):
        max_iter = 200
    else:  # balanced
        max_iter = 80

    for _ in range(max_iter):
        improved = False
        neighbour_best = None

        for uid in uids:
            current_val = best_lots.get(uid, 0)
            for delta in (-1, 1):
                new_val = current_val + delta
                if new_val < min_lot or new_val > max_lot:
                    continue

                cand_lots = dict(best_lots)
                cand_lots[uid] = new_val

                # Respect total lots cap
                if total_lots(cand_lots) > max_total_lots:
                    continue

                # Do not allow fully flat portfolio if baseline had exposure
                if baseline_total_lots > 0 and total_lots(cand_lots) == 0:
                    continue

                metrics_c, viol_c, rank_c = eval_lots(cand_lots)
                if metrics_c is None:
                    continue

                if rank_c < best_rank:
                    neighbour_best = (cand_lots, metrics_c, viol_c, rank_c)
                    best_rank = rank_c
                    improved = True

        if not improved or neighbour_best is None:
            break

        cand_lots, metrics_c, viol_c, rank_c = neighbour_best
        best_lots = cand_lots
        best_metrics = metrics_c
        best_viol = viol_c
        best_rank = rank_c

    # If final solution still violates constraints, treat as "no feasible"
    if best_viol and len(best_viol) > 0:
        return base_lots, None, ["no_feasible"]

    return best_lots, best_metrics, best_viol



def _render_metrics_compare(baseline, candidate, baseline_viol, candidate_viol):
    def fmt_money(x):
        if x is None:
            return "—"
        return f"${x:,.0f}"

    def fmt_int(x):
        if x is None:
            return "—"
        return f"{int(x)}"

    rows = [
        ("Total P&L", fmt_money(baseline.get("total_pnl")), fmt_money(candidate.get("total_pnl"))),
        ("Max DD", fmt_money(baseline.get("max_dd_abs")), fmt_money(candidate.get("max_dd_abs"))),
        ("Max daily loss", fmt_money(baseline.get("max_daily_loss")), fmt_money(candidate.get("max_daily_loss"))),
        ("Max losing streak (days)", fmt_int(baseline.get("max_losing_streak")), fmt_int(candidate.get("max_losing_streak"))),
        ("Max DD duration (days)", fmt_int(baseline.get("max_dd_duration")), fmt_int(candidate.get("max_dd_duration"))),
        ("Constraint violations", fmt_int(len(baseline_viol)), fmt_int(len(candidate_viol))),
    ]

    table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Metric", style={"textAlign": "left", "padding": "6px"}),
                        html.Th("Baseline", style={"textAlign": "right", "padding": "6px"}),
                        html.Th("Candidate", style={"textAlign": "right", "padding": "6px"}),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(a, style={"padding": "6px", "color": "#DDDDDD"}),
                            html.Td(b, style={"padding": "6px", "textAlign": "right"}),
                            html.Td(c, style={"padding": "6px", "textAlign": "right"}),
                        ],
                        style={"borderTop": "1px solid #333333"},
                    )
                    for a, b, c in rows
                ]
            ),
        ],
        style={"width": "100%", "borderCollapse": "collapse"},
    )

    viol_note = html.Div(
        [
            html.Div(f"Baseline violations: {', '.join(baseline_viol) if baseline_viol else 'None'}", style={"marginTop": "0.5rem", "color": "#AAAAAA", "fontSize": "0.75rem"}),
            html.Div(f"Candidate violations: {', '.join(candidate_viol) if candidate_viol else 'None'}", style={"color": "#AAAAAA", "fontSize": "0.75rem"}),
        ]
    )
    return html.Div([table, viol_note])


def _render_lots_table(uid_to_name, lots_map):
    df = pd.DataFrame(
        [{"Strategy": uid_to_name.get(uid, uid), "Lots": int(lots_map.get(uid, 0))} for uid in uid_to_name.keys()]
    )
    df = df.sort_values(["Lots", "Strategy"], ascending=[False, True])

    return html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Strategy", style={"textAlign": "left", "padding": "6px"}),
                        html.Th("Lots", style={"textAlign": "right", "padding": "6px"}),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(row["Strategy"], style={"padding": "6px", "color": "#DDDDDD"}),
                            html.Td(str(int(row["Lots"])), style={"padding": "6px", "textAlign": "right"}),
                        ],
                        style={"borderTop": "1px solid #333333"},
                    )
                    for _, row in df.iterrows()
                ]
            ),
        ],
        style={"width": "100%", "borderCollapse": "collapse"},
    )


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

@callback(
    Output("p2-builder-metrics-compare", "children"),
    Output("p2-builder-lots-table", "children"),
    Output("p2-builder-last-run", "data"),
    Output("p2-builder-apply-lots", "data", allow_duplicate=True),  # NEW target
    Output("p2-builder-status-local", "children"),
    Input("p2-builder-run-btn", "n_clicks"),
    Input("p2-builder-apply-btn", "n_clicks"),
    Input("p2-builder-discard-btn", "n_clicks"),
    State("p1-active-list-store", "data"),
    State("p2-weights-store", "data"),
    State("p2-builder-last-run", "data"),
    State("p2-builder-run-mode", "value"),
    State("p2-builder-initial-equity-input", "value"),
    State("p2-builder-max-total-lots", "value"),
    State("p2-builder-min-lots", "value"),
    State("p2-builder-max-lots", "value"),
    State("p2-builder-max-budget", "value"),
    State("p2-builder-max-daily-loss", "value"),
    State("p2-builder-max-dd", "value"),
    State("p2-builder-max-losing-streak", "value"),
    State("p2-builder-max-dd-duration", "value"),
    prevent_initial_call=True,
)

def run_builder(
    n_run,
    n_apply,
    n_discard,
    active_store,
    weights_store,
    last_run,
    run_mode,
    initial_equity,
    max_total_lots,
    min_lot,
    max_lot,
    max_budget,
    max_daily_loss,
    max_dd,
    max_losing_streak,
    max_dd_duration,
):
    trig = ctx.triggered_id

    # Never run on initial page load / layout build
    if trig is None:
        return no_update, no_update, no_update, no_update, no_update


    # -------------------------
    # Discard
    # -------------------------
    if trig == "p2-builder-discard-btn":
        return no_update, no_update, None, no_update, "Discarded. Preview cleared."

    # -------------------------
    # Apply
    # -------------------------
    if trig == "p2-builder-apply-btn":
        if not last_run or "candidate_lots" not in last_run:
            return no_update, no_update, no_update, no_update, "Nothing to apply (run Builder first)."

        lots = last_run["candidate_lots"]
        # Publish raw lots to the bridge store; weights panel will convert them to factors
        return no_update, no_update, last_run, lots, "Applied. Weights updated (factors = integer lots)."


    # -------------------------
    # Run Builder
    # -------------------------
    if trig == "p2-builder-run-btn":
        if not n_run:
            return no_update, no_update, no_update, no_update, no_update

        rows = _selected_active_rows(active_store)
        if not rows:
            return (
                html.Div("No selected strategies in Active list.", style={"color": "#FFAAAA"}),
                html.Div("—"),
                None,
                no_update,
                "No strategies selected.",
            )

        uid_to_name = _get_uid_name_map(active_store)
        uids = list(uid_to_name.keys())

        base_lots = _lots_from_weights(weights_store, uids, default_lot=1)

        constraints = {
            "max_total_lots": max_total_lots,
            "min_lot": min_lot,
            "max_lot": max_lot,
            "max_budget": max_budget,  # not yet enforced in v1
            "max_daily_loss": None if max_daily_loss in (None, "") else float(max_daily_loss),
            "max_dd": None if max_dd in (None, "") else float(max_dd),
            "max_losing_streak": None if max_losing_streak in (None, "") else int(max_losing_streak),
            "max_dd_duration": None if max_dd_duration in (None, "") else int(max_dd_duration),
        }

        # baseline metrics (current lots)
        w_base = _weights_from_lots({}, base_lots)
        ts_base = _portfolio_series(active_store, w_base, initial_equity, weight_mode="lots")
        m_base = _compute_key_metrics(ts_base) or {}
        v_base = _violations(m_base, constraints)
        
        # search for integer-lot portfolio under hard constraints
        cand_lots, m_cand, v_cand = _search_integer_lots(
            active_store=active_store,
            base_lots=base_lots,
            constraints=constraints,
            run_mode=run_mode,
            initial_equity=initial_equity,
        )
        
        # If no feasible / no-data candidate was found, keep previous results
        # and show a strong warning. IMPORTANT: we do *not* overwrite
        # metrics compare, lots table or last_run here.
        if m_cand is None:
            reason = v_cand[0] if isinstance(v_cand, list) and v_cand else "unknown"
            if reason == "no_feasible":
                msg = (
                    "Run complete: no portfolio satisfies all current constraints and lot bounds. "
                    "Displayed metrics and lots are from the last successful run (or baseline if none). "
                    "Relax constraints and run again."
                )
            elif reason == "no_data":
                msg = (
                    "Run complete: unable to compute metrics for this selection (no data). "
                    "Displayed metrics and lots are from the last successful run (or baseline if none)."
                )
            else:
                msg = (
                    "Run complete: no valid candidate portfolio could be constructed. "
                    "Displayed metrics and lots are from the last successful run (or baseline if none)."
                )
        
            status = html.Span(
                msg,
                style={
                    #"marginLeft": "1rem",
                    "fontSize": "0.85rem",
                    "color": "#FF6666",
                    "fontWeight": "bold",
                    "flex": "1 1 0",
                    "minWidth": 0,
                },
            )

            return no_update, no_update, last_run, no_update, status
        
        # Feasible candidate found
        m_cand = m_cand or {}
        v_cand = v_cand or []
        
        compare = _render_metrics_compare(m_base, m_cand, v_base, v_cand)
        lots_table = _render_lots_table(uid_to_name, cand_lots)
        
        payload = {
            "candidate_lots": cand_lots,
            "candidate_metrics": m_cand,
            "candidate_violations": v_cand,
        }
        
        # Build a meaningful status message
        if not v_cand:
            if v_base and not v_cand:
                status_text = (
                    "Run complete. Baseline violates one or more constraints; "
                    "candidate satisfies all and respects lot bounds."
                )
            else:
                # both baseline and candidate feasible
                pnl_base = float(m_base.get("total_pnl", 0.0) or 0.0)
                pnl_cand = float(m_cand.get("total_pnl", 0.0) or 0.0)
                if pnl_cand > pnl_base + 1e-6:
                    status_text = (
                        "Run complete. Candidate satisfies all constraints and increases total P&L "
                        "vs baseline."
                    )
                elif pnl_cand < pnl_base - 1e-6:
                    status_text = (
                        "Run complete. Candidate satisfies all constraints but has lower total P&L "
                        "than baseline; review before applying."
                    )
                else:
                    status_text = (
                        "Run complete. Candidate satisfies all constraints and is similar to baseline "
                        "on total P&L."
                    )

        else:
            # Should not occur because _search_integer_lots only returns feasible candidates,
            # but keep a fallback.
            status_text = (
                "Run complete. Candidate returned with constraint violations; please verify inputs."
            )
        
        status = html.Span(
                    status_text,
                    style={
                        "marginLeft": "1rem",
                        "fontSize": "0.85rem",
                        "color": "springgreen",   # or red for errors
                        "fontWeight": "bold",
                        "flex": "1 1 0",
                        "minWidth": 0,
                    },
                )

        
        return compare, lots_table, payload, no_update, status






