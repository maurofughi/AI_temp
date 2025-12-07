# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 10:18:10 2025

@author: mauro
"""

# -*- coding: utf-8 -*-
"""
Phase 2 – Portfolio Construction

Main page layout for Phase 2, reusing shared loader + strategy list.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, ctx, no_update

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core.sh_layout import (
    build_data_input_section,
    build_strategy_sidebar,
    ROOT_DATA_DIR,
    _list_immediate_subfolders,
    p1_strategy_store,   # reuse in-memory strategy data (with df)
)
from core.registry import list_portfolios



def build_phase2_right_panel():
    """
    Phase 2 right-hand panel (Portfolio Analytics / Allocation Scenarios / Optimizer).
    Does NOT include loader or strategy list. Reused later by main.py.
    """
    return dbc.Card(
        [
            dbc.CardHeader("Portfolio-level analysis and optimizer"),
            dbc.CardBody(
                [
                    dcc.Tabs(
                        id="p2-analysis-tabs",
                        value="p2-analytics",
                        style={"backgroundColor": "#222222"},
                        children=[
                            dcc.Tab(
                                label="Portfolio Analytics",
                                value="p2-analytics",
                                style={
                                    "backgroundColor": "#333333",
                                    "color": "#BBBBBB",
                                    "padding": "8px 12px",
                                },
                                selected_style={
                                    "backgroundColor": "#555555",
                                    "color": "#FFFFFF",
                                    "padding": "8px 12px",
                                },
                                children=html.Div(
                                    [
                                        # ------- Top controls + metrics ---------------------------------
                                        dbc.Row(
                                            [
                                                # Left: Initial equity input
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Span(
                                                                "Initial equity:",
                                                                style={
                                                                    "marginRight": "0.5rem",
                                                                    "fontSize": "0.85rem",
                                                                },
                                                            ),
                                                            dcc.Input(
                                                                id="p2-initial-equity-input",
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
                                                        style={
                                                            "display": "flex",
                                                            "alignItems": "center",
                                                            "gap": "0.5rem",
                                                        },
                                                    ),
                                                    md=5,
                                                ),
                                                # Right: Weighting mode selector
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Span(
                                                                "Weighting mode:",
                                                                style={
                                                                    "marginRight": "0.5rem",
                                                                    "fontSize": "0.85rem",
                                                                },
                                                            ),
                                                            dbc.RadioItems(
                                                                id="p2-weight-mode",
                                                                options=[
                                                                    {
                                                                        "label": "Factors (fractional)",
                                                                        "value": "factors",
                                                                    },
                                                                    {
                                                                        "label": "Integer lots (preview)",
                                                                        "value": "lots",
                                                                    },
                                                                ],
                                                                value="factors",
                                                                inline=True,
                                                                className="btn-group",
                                                                inputClassName="btn-check",
                                                                labelClassName="btn btn-sm btn-outline-info",
                                                                labelCheckedClassName="btn btn-sm btn-info active",
                                                            ),
                                                        ],
                                                        style={
                                                            "display": "flex",
                                                            "alignItems": "center",
                                                            "justifyContent": "flex-end",
                                                            "gap": "0.75rem",
                                                        },
                                                    ),
                                                    md=7,
                                                ),
                                            ],
                                            className="mb-2",
                                        ),
                                        # Metrics table / summary for portfolio
                                        html.Div(
                                            id="p2-portfolio-metrics",
                                            style={
                                                "fontSize": "0.85rem",
                                                "marginBottom": "0.75rem",
                                            },
                                        ),
                                        html.Hr(style={"marginTop": "0.25rem", "marginBottom": "0.75rem"}),
                            
                                        # ------- Middle: Equity & Drawdown charts (stacked full-width) --
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-equity-graph",
                                                        figure={
                                                            "data": [],
                                                            "layout": {
                                                                "template": "plotly_dark",
                                                                "paper_bgcolor": "#222222",
                                                                "plot_bgcolor": "#222222",
                                                                "font": {"color": "#EEEEEE"},
                                                            },
                                                        },
                                                        style={"height": "320px"},
                                                    ),
                                                    md=12,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-dd-graph",
                                                        figure={
                                                            "data": [],
                                                            "layout": {
                                                                "template": "plotly_dark",
                                                                "paper_bgcolor": "#222222",
                                                                "plot_bgcolor": "#222222",
                                                                "font": {"color": "#EEEEEE"},
                                                            },
                                                        },
                                                        style={"height": "260px"},
                                                    ),
                                                    md=12,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),

                            
                                        # ------- Bottom: Histogram + DOW exposure -----------------------
                                        dbc.Row(
                                            [
                                                # Left: P&L histogram + distribution metrics
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                id="p2-dist-metrics",
                                                                style={
                                                                    "fontSize": "0.8rem",
                                                                    "marginBottom": "0.4rem",
                                                                },
                                                            ),
                                                            dcc.Graph(
                                                                id="p2-pnl-histogram",
                                                                figure={
                                                                    "data": [],
                                                                    "layout": {
                                                                        "template": "plotly_dark",
                                                                        "paper_bgcolor": "#222222",
                                                                        "plot_bgcolor": "#222222",
                                                                        "font": {"color": "#EEEEEE"},
                                                                    },
                                                                },
                                                                style={"height": "260px"},
                                                            ),
                                                        ]
                                                    ),
                                                    md=6,
                                                ),
                                                # Right: Day-of-week exposure chart
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-dow-bar-graph",
                                                        figure={
                                                            "data": [],
                                                            "layout": {
                                                                "template": "plotly_dark",
                                                                "paper_bgcolor": "#222222",
                                                                "plot_bgcolor": "#222222",
                                                                "font": {"color": "#EEEEEE"},
                                                            },
                                                        },
                                                        style={"height": "260px"},
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="mb-2",
                                        ),
                                    ],
                                    style={
                                        "padding": "0.75rem",
                                        "fontSize": "0.85rem",
                                    },
                                ),
                            ),

                            dcc.Tab(
                                label="Allocation Scenarios",
                                value="p2-scenarios",
                                children=html.Div(
                                    "Phase 2 – Allocation Scenarios (placeholder)",
                                    style={"padding": "1rem", "fontSize": "0.9rem"},
                                ),
                                style={
                                    "backgroundColor": "#333333",
                                    "color": "#BBBBBB",
                                    "padding": "8px 12px",
                                },
                                selected_style={
                                    "backgroundColor": "#555555",
                                    "color": "#FFFFFF",
                                    "padding": "8px 12px",
                                },
                            ),
                            dcc.Tab(
                                label="Portfolio Optimizer",
                                value="p2-optimizer",
                                children=html.Div(
                                    "Phase 2 – Portfolio Optimizer (placeholder)",
                                    style={"padding": "1rem", "fontSize": "0.9rem"},
                                ),
                                style={
                                    "backgroundColor": "#333333",
                                    "color": "#BBBBBB",
                                    "padding": "8px 12px",
                                },
                                selected_style={
                                    "backgroundColor": "#555555",
                                    "color": "#FFFFFF",
                                    "padding": "8px 12px",
                                },
                            ),
                        ],
                    )
                ]
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Phase 2 – internal helpers for portfolio analytics
# ---------------------------------------------------------------------------


def _metric_cell(label: str, value: str) -> html.Div:
    """
    Small metric cell styled similarly to Phase-1 header metrics.
    """
    return html.Div(
        [
            html.Div(label, style={"fontSize": "0.75rem", "color": "#AAAAAA"}),
            html.Div(
                value,
                style={
                    "fontSize": "0.95rem",
                    "fontWeight": "bold",
                    "marginTop": "0.15rem",
                },
            ),
        ],
        style={
            "padding": "0.35rem 0.6rem",
            "borderRadius": "4px",
            "backgroundColor": "#2b2b2b",
            "border": "1px solid #444444",
            "marginRight": "0.5rem",
            "marginBottom": "0.4rem",
            "minWidth": "120px",
        },
    )


def _losing_streak_stats(daily_pnl: pd.Series) -> tuple[int, float]:
    """
    Compute worst and average losing streak (consecutive negative days).
    Only streaks with length >= 2 are counted for the average.
    """
    max_streak = 0
    streaks = []

    current = 0
    for v in daily_pnl:
        if v < 0:
            current += 1
        else:
            if current >= 2:
                streaks.append(current)
                max_streak = max(max_streak, current)
            current = 0

    # handle trailing streak
    if current >= 2:
        streaks.append(current)
        max_streak = max(max_streak, current)

    if streaks:
        avg_streak = float(np.mean(streaks))
    else:
        avg_streak = 0.0
        max_streak = 0

    return max_streak, avg_streak


def _build_portfolio_timeseries(
    active_store: list,
    weights_store: dict | None,
    weight_mode: str,
    initial_equity: float,
) -> dict | None:
    """
    Core engine: take selected strategies from Active list, their df's from
    p1_strategy_store, and compute portfolio daily P&L, equity and DD.

    - P&L date = 'Date Closed' (summed by day)
    - weight_mode: 'factors' or 'lots'
    - factors = size multipliers (fractional lots)
    - lots_vec = integer lots in 'lots' mode
    - P&L is always ADDITIVE: Σ lots_i * P&L_i (no normalisation)
    """

    active_store = active_store or []

    # Use only SELECTED strategies in Active list
    selected_rows = [r for r in active_store if r.get("is_selected")]
    if not selected_rows:
        return None

    # Build per-strategy daily P&L series
    pnl_series_by_uid: dict[str, pd.Series] = {}

    for row in selected_rows:
        uid = row.get("uid")
        if not uid:
            continue

        meta = p1_strategy_store.get(uid, {})
        df = meta.get("df")
        if df is None:
            continue

        if "Date Closed" not in df.columns or "P/L" not in df.columns:
            continue

        tmp = df[["Date Closed", "P/L"]].copy()
        tmp["Date Closed"] = pd.to_datetime(tmp["Date Closed"]).dt.date
        s = tmp.groupby("Date Closed")["P/L"].sum()

        if not s.empty:
            pnl_series_by_uid[uid] = s

    if not pnl_series_by_uid:
        return None

    # Align all series onto a common daily index
    all_dates = sorted(set().union(*[s.index for s in pnl_series_by_uid.values()]))
    idx = pd.to_datetime(all_dates)

    pnl_df = pd.DataFrame(index=idx)
    for uid, s in pnl_series_by_uid.items():
        pnl_df[uid] = s.reindex(all_dates, fill_value=0.0).values

    active_uids = list(pnl_df.columns)
    n_strats = len(active_uids)

    # ---------------- factors -> lots_vec (size) ----------------
    weights_store = weights_store or {}

    # base factors from store (>=0, fallback 1.0)
    factors = []
    for uid in active_uids:
        entry = weights_store.get(uid, {}) if isinstance(weights_store, dict) else {}
        f = entry.get("factor", 1.0)
        try:
            f = float(f)
        except (TypeError, ValueError):
            f = 1.0
        if f < 0:
            f = 0.0
        factors.append(f)

    factors = np.array(factors, dtype=float)

    # lots_vec is what actually scales P&L
    if weight_mode == "lots":
        # Integer lots preview:
        #  - factor <= 0  -> 0 lots
        #  - factor > 0   -> at least 1 lot (round, then floor to 1)
        lots_vec = np.zeros_like(factors, dtype=int)
        positive_mask = factors > 0
        if positive_mask.any():
            rounded = np.round(factors[positive_mask]).astype(int)
            rounded[rounded < 1] = 1
            lots_vec[positive_mask] = rounded
        # fallback if everything is 0 for some reason
        if lots_vec.sum() == 0 and positive_mask.any():
            lots_vec[positive_mask] = 1
    else:
        # 'factors' mode: use fractional factors directly as size multipliers
        lots_vec = factors.copy()

    # Derived weights (for display/allocation only)
    total_size = float(lots_vec.sum())
    if total_size > 0:
        weights = pd.Series(lots_vec / total_size, index=active_uids)
    else:
        weights = pd.Series(
            np.ones(n_strats) / max(n_strats, 1),
            index=active_uids,
        )

    # ---------------- portfolio series (ADDITIVE) ----------------
    portfolio_daily = pd.Series(
        pnl_df.values @ lots_vec,
        index=pnl_df.index,
        name="portfolio_daily_pnl",
    )

    initial_equity = float(initial_equity) if initial_equity is not None else 100000.0
    equity = pd.Series(initial_equity + portfolio_daily.cumsum(), index=pnl_df.index)
    running_max = equity.cummax()
    dd = running_max - equity
    dd_pct = dd / running_max.replace(0, np.nan)

    return {
        "dates": pnl_df.index,
        "pnl_df": pnl_df,
        "portfolio_daily": portfolio_daily,
        "equity": equity,
        "dd": dd,
        "dd_pct": dd_pct,
        "weights": weights,      # allocation %
        "lots_vec": lots_vec,    # actual size multipliers used
        "initial_equity": initial_equity,
    }



# ---------------------------------------------------------------------------
# Phase 2 – Portfolio Analytics callback
# ---------------------------------------------------------------------------


@callback(
    Output("p2-portfolio-metrics", "children"),
    Output("p2-equity-graph", "figure"),
    Output("p2-dd-graph", "figure"),
    Output("p2-dist-metrics", "children"),
    Output("p2-pnl-histogram", "figure"),
    Output("p2-dow-bar-graph", "figure"),
    Input("p1-active-list-store", "data"),
    Input("p2-weights-store", "data"),
    Input("p2-weight-mode", "value"),
    Input("p2-initial-equity-input", "value"),
)
def update_portfolio_analytics(
    active_store,
    weights_store,
    weight_mode,
    initial_equity,
):
    # Compute portfolio series
    series = _build_portfolio_timeseries(
        active_store=active_store,
        weights_store=weights_store,
        weight_mode=weight_mode or "factors",
        initial_equity=initial_equity or 100000.0,
    )

    if series is None:
        # No data – return empty-ish figures but keep layout
        empty_fig = {
            "data": [],
            "layout": {
                "template": "plotly_dark",
                "paper_bgcolor": "#222222",
                "plot_bgcolor": "#222222",
                "font": {"color": "#EEEEEE"},
            },
        }
        msg = html.Div(
            "No selected strategies with data – select strategies in Active list.",
            style={"fontSize": "0.8rem", "color": "#AAAAAA"},
        )
        return msg, empty_fig, empty_fig, msg, empty_fig, empty_fig

    dates = series["dates"]
    portfolio_daily = series["portfolio_daily"]
    equity = series["equity"]
    dd = series["dd"]
    dd_pct = series["dd_pct"]
    weights = series["weights"]
    initial_equity = series["initial_equity"]

    # ------------------------------------------------------------------
    # Metrics summary
    # ------------------------------------------------------------------
    n_days = len(portfolio_daily)
    if n_days == 0:
        total_pnl = max_dd_abs = max_dd_pct = 0.0
        sharpe_ann = cagr = mar = win_rate = 0.0
        avg_daily_pnl = avg_monthly_pnl = 0.0
        worst_ls, avg_ls = 0, 0.0
        top5_pct = float("nan")
    else:
        total_pnl = float(equity.iloc[-1] - initial_equity)
        max_dd_abs = float(dd.max())
        max_dd_pct = float((dd_pct.max() or 0.0) * 100.0)

        # Daily return based on initial equity, same as Phase 1 logic
        daily_ret = portfolio_daily / initial_equity
        mu = daily_ret.mean()
        sigma = daily_ret.std(ddof=1)
        sharpe_ann = float(mu / sigma * np.sqrt(252.0)) if sigma > 0 else 0.0

        # Years based on calendar span between first and last date
        if len(dates) > 1:
            years = max((dates[-1] - dates[0]).days / 365.25, 1e-6)
        else:
            years = max(n_days / 252.0, 1e-6)

        cagr = float((equity.iloc[-1] / initial_equity) ** (1.0 / years) - 1.0)
        mar = float(cagr / (abs(max_dd_pct) / 100.0)) if max_dd_pct > 0 else 0.0

        win_rate = float((portfolio_daily > 0).mean() * 100.0)
        avg_daily_pnl = float(portfolio_daily.mean())

        monthly_pnl = portfolio_daily.resample("ME").sum()
        avg_monthly_pnl = float(monthly_pnl.mean()) if not monthly_pnl.empty else 0.0

        worst_ls, avg_ls = _losing_streak_stats(portfolio_daily)

        # Concentration: % of total P&L coming from top 5% days
        if n_days >= 5 and abs(total_pnl) > 0:
            q = int(max(1, np.floor(n_days * 0.05)))
            top5_sum = float(portfolio_daily.nlargest(q).sum())
            top5_pct = float(top5_sum / total_pnl * 100.0)
        else:
            top5_pct = float("nan")
            
        # Gini & HHI of positive-day contributions (wins only)
        pos = portfolio_daily[portfolio_daily > 0]
        if len(pos) >= 2 and pos.sum() > 0:
            shares = (pos / pos.sum()).values  # each win's fraction of total wins P&L

            # HHI: sum of squared shares
            hhi = float(np.sum(shares ** 2))

            # Gini: standard discrete formula on shares
            shares_sorted = np.sort(shares)
            n_pos = len(shares_sorted)
            index = np.arange(1, n_pos + 1)
            gini = float(
                np.sum((2 * index - n_pos - 1) * shares_sorted)
                / (n_pos * np.sum(shares_sorted))
            )
        else:
            gini = float("nan")
            hhi = float("nan")

    metrics_bar = html.Div(
        [
            _metric_cell("Total P&L", f"${total_pnl:,.0f}"),
            _metric_cell("Max DD ($)", f"${max_dd_abs:,.0f}"),
            _metric_cell("Max DD (%)", f"{max_dd_pct:,.2f}%"),
            _metric_cell("CAGR", f"{cagr * 100:,.2f}%"),
            _metric_cell("MAR", f"{mar:,.2f}"),
            _metric_cell("Sharpe (ann.)", f"{sharpe_ann:,.2f}"),
            _metric_cell("Win rate (days)", f"{win_rate:,.1f}%"),
            _metric_cell("Avg daily P&L", f"${avg_daily_pnl:,.0f}"),
            _metric_cell("Avg monthly P&L", f"${avg_monthly_pnl:,.0f}"),
            _metric_cell("Worst losing streak", f"{worst_ls} days"),
            _metric_cell(
                "Avg losing streak (>=2)", f"{avg_ls:,.1f} days"
            ),
            _metric_cell(
                "Top-5% days P&L",
                "N/A" if np.isnan(top5_pct) else f"{top5_pct:,.1f}%",
            ),
            _metric_cell(
                "Gini (wins)",
                "N/A" if np.isnan(gini) else f"{gini:,.2f}",
            ),
            _metric_cell(
                "HHI (wins)",
                "N/A" if np.isnan(hhi) else f"{hhi:,.3f}",
            ),
        ],
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "alignItems": "flex-start",
        },
    )


    # ------------------------------------------------------------------
    # Equity and DD figures
    # ------------------------------------------------------------------
    equity_fig = go.Figure()
    equity_fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity,
            mode="lines",
            name="Portfolio equity",
        )
    )
    equity_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin=dict(l=40, r=10, t=25, b=40),
        xaxis_title="Date",
        yaxis_title="Equity",
    )

    dd_fig = go.Figure()
    dd_fig.add_trace(
        go.Scatter(
            x=dates,
            y=-dd,  # plot as negative for visual convention
            mode="lines",
            name="Drawdown",
        )
    )
    dd_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin=dict(l=40, r=10, t=25, b=40),
        xaxis_title="Date",
        yaxis_title="Drawdown",
    )

    # ------------------------------------------------------------------
    # Distribution metrics + histogram
    # ------------------------------------------------------------------
    skew = float(portfolio_daily.skew()) if n_days > 1 else 0.0
    kurt = float(portfolio_daily.kurtosis()) if n_days > 1 else 0.0

    # Tail ratio: mean of top 5% / abs(mean of bottom 5%)
    if n_days > 10:
        q = int(max(1, np.floor(n_days * 0.05)))
        top = portfolio_daily.nlargest(q).mean()
        bottom = portfolio_daily.nsmallest(q).mean()
        tail_ratio = float(top / abs(bottom)) if bottom < 0 else np.nan
    else:
        tail_ratio = np.nan

    # # % P&L from top 5 days
    # if n_days >= 5 and total_pnl != 0:
    #     q = int(max(1, np.floor(n_days * 0.05)))
    #     top5_sum = float(portfolio_daily.nlargest(q).sum())
    #     top5_pct = float(top5_sum / total_pnl * 100.0)        
    # else:
    #     top5_pct = np.nan

    dist_metrics = html.Div(
        [
            html.Span(f"Skewness: {skew:,.2f}", style={"marginRight": "1rem"}),
            html.Span(f"Kurtosis: {kurt:,.2f}", style={"marginRight": "1rem"}),
            html.Span(
                "Tail ratio: N/A" if np.isnan(tail_ratio) else f"Tail ratio: {tail_ratio:,.2f}",
                style={"marginRight": "1rem"},
            ),
            html.Span(
                "Top-5% days P&L: N/A"
                if np.isnan(top5_pct)
                else f"Top-5% days P&L: {top5_pct:,.1f}%",
            ),
        ],
        style={"fontSize": "0.8rem"},
    )

    hist_fig = go.Figure()
    hist_fig.add_trace(
        go.Histogram(
            x=portfolio_daily,
            nbinsx=40,
            name="Daily P&L",
        )
    )
    hist_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin=dict(l=40, r=10, t=25, b=40),
        xaxis_title="Daily P&L",
        yaxis_title="Count",
        showlegend=False,
    )

    # ------------------------------------------------------------------
    # Day-of-week exposure bar
    # ------------------------------------------------------------------
    dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    dow = portfolio_daily.copy()
    dow.index = pd.DatetimeIndex(dow.index)
    dow_group = dow.groupby(dow.index.dayofweek).sum()

    dow_x = [dow_map.get(i, str(i)) for i in dow_group.index]
    dow_y = [float(v) for v in dow_group.values]

    dow_fig = go.Figure()
    dow_fig.add_trace(
        go.Bar(
            x=dow_x,
            y=dow_y,
            name="Total P&L by weekday",
        )
    )
    dow_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin=dict(l=40, r=10, t=25, b=40),
        xaxis_title="Day of week",
        yaxis_title="Total P&L",
        showlegend=False,
    )

    return metrics_bar, equity_fig, dd_fig, dist_metrics, hist_fig, dow_fig




def layout_page_2_portfolio():
    """
    Layout for Phase 2 – Portfolio Construction.

    Left/top:
        - Shared 'Data Input – Strategies' collapsible panel
        - Shared 'Strategy List' sidebar

    Right:
        - Phase 2-specific tabs (Portfolio Analytics, placeholder, Optimizer)
    """

    # ------------------------------------------------------------------
    # Folder list + checklist (same idea as Phase 1)
    # ------------------------------------------------------------------
    subfolders = _list_immediate_subfolders(ROOT_DATA_DIR)

    if not subfolders:
        folder_info = html.Div(
            [
                html.Div(
                    f"No subfolders found under root data folder:",
                    style={"fontSize": "0.85rem"},
                ),
                html.Code(ROOT_DATA_DIR, style={"fontSize": "0.8rem"}),
            ],
            style={"marginBottom": "0.5rem"},
        )
        folder_checklist = html.Div(
            "No folders available for scanning.",
            style={"fontSize": "0.8rem", "color": "#AAAAAA"},
        )
    else:
        folder_info = html.Div(
            [
                html.Div(
                    "Root data folder for strategies:",
                    style={"fontSize": "0.85rem"},
                ),
                html.Code(ROOT_DATA_DIR, style={"fontSize": "0.8rem"}),
                html.Div(
                    "Select one or more subfolders to scan for CSV strategy files.",
                    style={"fontSize": "0.8rem", "marginTop": "0.25rem"},
                ),
            ],
            style={"marginBottom": "0.5rem"},
        )

        # Checklist options: show only folder name in label, full path as value
        folder_checklist = dcc.Checklist(
            id="p1-folder-checklist",
            options=[
                {"label": name, "value": full_path} for name, full_path in subfolders
            ],
            value=[],
            labelStyle={"display": "block"},
            style={"fontSize": "0.8rem", "maxHeight": "200px", "overflowY": "auto"},
        )

    # ------------------------------------------------------------------
    # Portfolio dropdown options (same registry as Phase 1)
    # ------------------------------------------------------------------
    portfolios = list_portfolios()
    portfolio_options = [
        {
            "label": p.get("name", p.get("id", "Unnamed portfolio")),
            "value": p.get("id"),
        }
        for p in portfolios
        if p.get("id") is not None
    ]

    # ------------------------------------------------------------------
    # Assemble layout
    # ------------------------------------------------------------------
    return dbc.Container(
        [
            html.H3("Phase 2 – Portfolio Construction", className="mb-3"),

            # We still rely on shared p1-* IDs for loader + strategy list,
            # so no new Store is needed here for selection; Phase 2 analytics
            # will read from the same components as Phase 1.
            # If we need Phase-2-specific state later, we can add p2-* stores.

            # Shared portfolio identity store (needed by shared callbacks)
            dcc.Store(id="p1-current-portfolio-id", data=None),
            
            # Shared Data Input section (collapsible)
            build_data_input_section(folder_info, folder_checklist, portfolio_options),

            html.Hr(),

            dbc.Row(
                [
                    # LEFT: shared Strategy List sidebar
                    build_strategy_sidebar(),

                    # RIGHT: Phase 2 tabs
                    dbc.Col(
                        build_phase2_right_panel(),
                        width=9,
                    ),

                ]
            ),
        ],
        fluid=True,
    )
