# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 07:30:23 2025

@author: mauro
"""

import numpy as np
import pandas as pd
from statistics import NormalDist  # for BCa


from dash import callback, Input, Output, State, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from pages.page2 import _build_portfolio_timeseries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_robustness_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": "#AAAAAA", "size": 12},
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _extract_portfolio_pnl(active_store, weights_store, initial_equity_value):
    """
    Build portfolio timeseries via _build_portfolio_timeseries and return:
      - pnl: pd.Series of daily portfolio P&L in dollars
      - equity0: float, initial equity used

    Returns (None, None) if it cannot build.
    """
    active_store = active_store or []
    weights_store = weights_store or {}

    # At least one selected strategy
    selected_rows = [r for r in active_store if r.get("is_selected")]
    if len(selected_rows) == 0:
        return None, None

    try:
        base_initial = float(initial_equity_value or 100000.0)
    except Exception:
        base_initial = 100000.0

    try:
        series = _build_portfolio_timeseries(
            active_store=active_store,
            weights_store=weights_store,
            weight_mode="factors",      # same as main analytics metrics
            initial_equity=base_initial,
        )
    except Exception:
        return None, None

    if not series:
        return None, None

    portfolio_daily = series.get("portfolio_daily")
    initial_equity = float(series.get("initial_equity", base_initial))

    if portfolio_daily is None or len(portfolio_daily) < 10:
        return None, None

    pnl = portfolio_daily.astype(float).dropna()
    if len(pnl) < 10:
        return None, None

    return pnl, initial_equity


def _equity_from_pnl(pnl: np.ndarray, eq0: float) -> np.ndarray:
    """Build additive equity curve from daily P&L and initial equity."""
    return eq0 + np.cumsum(pnl)


def _sharpe_from_pnl(pnl: np.ndarray, eq0: float) -> float:
    """
    Sharpe based on daily returns defined as pnl / eq0 (constant denominator).
    """
    if pnl.size < 2 or eq0 <= 0:
        return 0.0
    ret = pnl / eq0
    mu = float(ret.mean())
    sigma = float(ret.std(ddof=1))
    if sigma <= 0:
        return 0.0
    return float(mu / sigma * np.sqrt(252.0))


def _max_dd_from_pnl(pnl: np.ndarray, eq0: float) -> tuple[float, float]:
    """
    Max drawdown from additive equity curve.

    Returns:
      (max_dd_abs, max_dd_pct) where:
        - max_dd_abs is in dollars
        - max_dd_pct is a fraction (0.2 = 20%)
    """
    if pnl.size == 0:
        return 0.0, 0.0
    equity = _equity_from_pnl(pnl, eq0)
    running_max = np.maximum.accumulate(equity)
    dd = running_max - equity
    max_dd_abs = float(dd.max())
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct_series = np.where(running_max > 0, dd / running_max, 0.0)
    max_dd_pct = float(np.nanmax(dd_pct_series))
    if not np.isfinite(max_dd_pct):
        max_dd_pct = 0.0
    return max_dd_abs, max_dd_pct


def _max_block_loss_from_pnl(pnl: np.ndarray, block_len: int, eq0: float) -> float:
    """
    Worst block P&L over a rolling window of length block_len,
    expressed as a fraction of initial equity (negative = loss).
    """
    n = pnl.size
    if n == 0 or eq0 <= 0:
        return 0.0
    block_len = max(1, int(block_len))
    if block_len > n:
        block_len = n

    worst = 0.0
    first = True
    for i in range(0, n - block_len + 1):
        window = pnl[i : i + block_len]
        block_pnl = float(window.sum())
        block_ret = block_pnl / eq0
        if first or block_ret < worst:
            worst = block_ret
            first = False
    return worst


def _block_bootstrap_sample_pnl(pnl: np.ndarray, block_len: int, target_len: int, rng) -> np.ndarray:
    """
    Block bootstrap on daily P&L: draw blocks of length block_len until target_len.

    Note: block_len = 1 => classic iid / non-block bootstrap.
    """
    n = pnl.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    block_len = max(1, int(block_len))
    samples = []

    while len(samples) < target_len:
        start = rng.integers(0, n)
        end = start + block_len
        if end <= n:
            block = pnl[start:end]
        else:
            overflow = end - n
            block = np.concatenate([pnl[start:], pnl[:overflow]])
        samples.append(block)

    resampled = np.concatenate(samples)[:target_len]
    return resampled.astype(float)



def _bca_interval_maxdd(
    pnl: np.ndarray,
    eq0: float,
    boot_maxdd: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    BCa confidence interval for Max DD (fraction, e.g. 0.2 = 20%).

    Parameters
    ----------
    pnl : np.ndarray
        Original daily P&L series.
    eq0 : float
        Initial equity.
    boot_maxdd : np.ndarray
        Bootstrap distribution of max drawdown fractions (same units as
        _max_dd_from_pnl second return value).
    alpha : float
        1 - confidence level (0.05 => 95% interval).

    Returns
    -------
    (low, high) : tuple[float, float]
        BCa lower / upper bounds in *fraction* units.
    """
    boot = np.asarray(boot_maxdd, dtype=float)
    boot = boot[np.isfinite(boot)]
    if boot.size == 0 or pnl.size < 5 or eq0 <= 0:
        # fall back to point estimate only
        _, theta_hat = _max_dd_from_pnl(pnl, eq0)
        return theta_hat, theta_hat

    # Observed statistic on full series
    _, theta_hat = _max_dd_from_pnl(pnl, eq0)

    # ----- Bias-correction term z0 -----
    prop_less = float(np.mean(boot < theta_hat))
    # clamp away from exactly 0 or 1
    eps = 1.0 / (2.0 * boot.size)
    prop_less = min(max(prop_less, eps), 1.0 - eps)

    nd = NormalDist()
    z0 = nd.inv_cdf(prop_less)

    # ----- Acceleration a via jackknife -----
    n = pnl.size
    theta_j = np.empty(n, dtype=float)
    for i in range(n):
        jack = np.delete(pnl, i)
        _, theta_j[i] = _max_dd_from_pnl(jack, eq0)

    theta_dot = float(theta_j.mean())
    num = float(np.sum((theta_dot - theta_j) ** 3))
    denom = float(6.0 * (np.sum((theta_dot - theta_j) ** 2) ** 1.5))
    a = num / denom if denom != 0.0 else 0.0

    def _bca_alpha(alpha_level: float) -> float:
        z_alpha = nd.inv_cdf(alpha_level)
        adj = z0 + (z0 + z_alpha) / (1.0 - a * (z0 + z_alpha))
        return nd.cdf(adj)

    alpha1 = alpha / 2.0
    alpha2 = 1.0 - alpha / 2.0
    p1 = _bca_alpha(alpha1)
    p2 = _bca_alpha(alpha2)

    low = float(np.quantile(boot, p1))
    high = float(np.quantile(boot, p2))
    return low, high


def _bootstrap_distribution_from_pnl(pnl: pd.Series, n_sim: int, block_len: int, eq0: float):
    ...


def _bootstrap_distribution_from_pnl(pnl: pd.Series, n_sim: int, block_len: int, eq0: float):
    """
    Block bootstrap distribution based on daily P&L.
    Returns arrays:
      sharpe_vals, maxdd_pct_vals, final_ret_vals, worst_block_pct_vals, final_equity_vals
    """
    rng = np.random.default_rng()
    pnl_arr = pnl.values.astype(float)
    n = pnl_arr.shape[0]

    n_sim = int(max(1, n_sim))
    block_len = int(max(1, block_len))

    sharpe_vals = np.empty(n_sim, dtype=float)
    maxdd_pct_vals = np.empty(n_sim, dtype=float)
    final_ret_vals = np.empty(n_sim, dtype=float)
    worst_block_pct_vals = np.empty(n_sim, dtype=float)
    final_equity_vals = np.empty(n_sim, dtype=float)

    for i in range(n_sim):
        sample_pnl = _block_bootstrap_sample_pnl(pnl_arr, block_len, n, rng)
        sharpe_vals[i] = _sharpe_from_pnl(sample_pnl, eq0)
        max_dd_abs, max_dd_pct = _max_dd_from_pnl(sample_pnl, eq0)
        maxdd_pct_vals[i] = max_dd_pct
        final_equity = float(eq0 + sample_pnl.sum())
        final_equity_vals[i] = final_equity
        final_ret_vals[i] = final_equity / eq0 - 1.0 if eq0 > 0 else 0.0
        worst_block_pct_vals[i] = _max_block_loss_from_pnl(sample_pnl, block_len, eq0)

    return sharpe_vals, maxdd_pct_vals, final_ret_vals, worst_block_pct_vals, final_equity_vals



def _mc_distribution_empirical(
    pnl_series: pd.Series,
    n_sim: int,
    horizon_days: int,
    eq0: float,
    ruin_threshold: float,
    max_paths_to_store: int = 200,
):
    """
    Empirical MC on daily P&L: resample historical daily P&L with replacement.

    Returns:
      - final_equity_vals: array of final equity values
      - maxdd_vals: array of max drawdown fractions
      - sharpe_vals: array of Sharpe ratios per path
      - ruin_flags: boolean array: True if equity ever drops below ruin_threshold
      - stored_paths: list of first max_paths_to_store equity paths (for fan chart)
      - min_equity_vals: array of minimum equity for ALL paths
      - worst_1d_loss_vals: array of worst single-day P&L per path
      - worst_5d_loss_vals: array of worst 5-day cumulative P&L per path
      - worst_20d_loss_vals: array of worst 20-day cumulative P&L per path
      - n_days_used: number of simulated days per path
    """
    pnl_arr = pnl_series.values.astype(float)
    n_hist = pnl_arr.shape[0]

    if n_hist == 0 or eq0 <= 0:
        empty = np.array([])
        return empty, empty, empty, empty, [], empty, empty, empty, empty, 0

    if horizon_days and horizon_days > 0:
        n_days = int(horizon_days)
    else:
        n_days = n_hist

    n_sim = max(1, int(n_sim))
    rng = np.random.default_rng()

    final_equity_vals = np.empty(n_sim, dtype=float)
    maxdd_vals = np.empty(n_sim, dtype=float)
    sharpe_vals = np.empty(n_sim, dtype=float)
    ruin_flags = np.empty(n_sim, dtype=bool)
    min_equity_vals = np.empty(n_sim, dtype=float)
    worst_1d_loss_vals = np.empty(n_sim, dtype=float)
    worst_5d_loss_vals = np.empty(n_sim, dtype=float)
    worst_20d_loss_vals = np.empty(n_sim, dtype=float)

    stored_paths: list[np.ndarray] = []

    for i in range(n_sim):
        # sample daily P&L
        path_pnl = rng.choice(pnl_arr, size=n_days, replace=True)

        # equity path
        equity = eq0 + np.cumsum(path_pnl)
        final_equity_vals[i] = equity[-1]

        # minimum equity over the path
        min_e = equity.min()
        min_equity_vals[i] = min_e

        # max drawdown from equity
        peak = np.maximum.accumulate(equity)
        dd_frac = 1.0 - equity / peak
        maxdd_vals[i] = dd_frac.max()

        # Sharpe (approx, using returns vs initial equity)
        path_ret = path_pnl / eq0
        mu = path_ret.mean()
        sigma = path_ret.std(ddof=1)
        sharpe_vals[i] = mu / sigma * np.sqrt(252) if sigma > 0 else 0.0

        # ruin flag: equity ever below threshold
        ruin_flags[i] = (min_e < ruin_threshold)

        # worst 1-day loss (most negative daily P&L)
        worst_1d_loss_vals[i] = path_pnl.min()

        # worst rolling 5-day and 20-day cumulative loss
        # (if horizon shorter than window, use full-sum as fallback)
        if n_days >= 5:
            roll5 = np.convolve(path_pnl, np.ones(5), mode="valid")
            worst_5d_loss_vals[i] = roll5.min()
        else:
            worst_5d_loss_vals[i] = path_pnl.sum()

        if n_days >= 20:
            roll20 = np.convolve(path_pnl, np.ones(20), mode="valid")
            worst_20d_loss_vals[i] = roll20.min()
        else:
            worst_20d_loss_vals[i] = path_pnl.sum()

        # store equity path for fan chart
        if len(stored_paths) < max_paths_to_store:
            stored_paths.append(equity)

    return (
        final_equity_vals,
        maxdd_vals,
        sharpe_vals,
        ruin_flags,
        stored_paths,
        min_equity_vals,
        worst_1d_loss_vals,
        worst_5d_loss_vals,
        worst_20d_loss_vals,
        n_days,
    )



def _hist_figure(
    data: np.ndarray,
    title: str,
    x_title: str,
    orig_value: float | None = None,
    orig_hover: str | None = None,
    p05: float | None = None,
    p95: float | None = None,
    bca_low: float | None = None,
    bca_high: float | None = None,
) -> go.Figure:
    """
    Generic histogram with:
      - red 'Actual' vline + marker
      - orange 5/95 percentile vlines
      - OPTIONAL fuchsia BCa vlines (low / high)
    """
    fig = go.Figure()

    if data.size > 0:
        fig.add_trace(go.Histogram(x=data, nbinsx=40))

    # Actual value
    if orig_value is not None:
        fig.add_vline(
            x=float(orig_value),
            line_color="red",
            line_width=2,
            annotation_text="Actual",
            annotation_position="top left",
        )
        hover_text = orig_hover or f"Actual: {orig_value:.3f}"
        fig.add_trace(
            go.Scatter(
                x=[float(orig_value)],
                y=[0],
                mode="markers",
                marker=dict(color="red", size=8),
                name="Actual",
                hovertemplate=hover_text + "",
                showlegend=False,
            )
        )

    # 5 / 95 percentiles (naive)
    if p05 is not None:
        fig.add_vline(
            x=float(p05),
            line_color="orange",
            line_width=1,
            line_dash="dash",
            annotation_text="5%",
            annotation_position="bottom left",
        )
    if p95 is not None:
        fig.add_vline(
            x=float(p95),
            line_color="orange",
            line_width=1,
            line_dash="dash",
            annotation_text="95%",
            annotation_position="bottom right",
        )

    # BCa interval (fuchsia)
    if bca_low is not None:
        fig.add_vline(
            x=float(bca_low),
            line_color="magenta",
            line_width=1,
            line_dash="dot",
            annotation_text="BCa low",
            annotation_position="top left",
        )
    if bca_high is not None:
        fig.add_vline(
            x=float(bca_high),
            line_color="magenta",
            line_width=1,
            line_dash="dot",
            annotation_text="BCa high",
            annotation_position="top right",
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin=dict(l=40, r=20, t=40, b=50),
        title=title,
        xaxis_title=x_title,
        yaxis_title="Count",
    )
    return fig



def _ecdf_figure(
    data: np.ndarray,
    orig_value: float,
    title: str,
    x_title: str,
    orig_hover: str | None = None,
) -> go.Figure:
    """
    ECDF figure with red vline and marker at original value.
    """
    fig = go.Figure()
    if data.size > 0:
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="ECDF",
            )
        )
        fig.add_vline(
            x=float(orig_value),
            line_color="red",
            line_width=2,
            annotation_text="Actual",
            annotation_position="top left",
        )
        hover_text = orig_hover or f"Actual: {orig_value:.3f}"
        fig.add_trace(
            go.Scatter(
                x=[float(orig_value)],
                y=[0.5],
                mode="markers",
                marker=dict(color="red", size=8),
                name="Actual",
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin=dict(l=40, r=20, t=40, b=50),
        title=title,
    )
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text="Cumulative probability", range=[0.0, 1.0])
    return fig


def _metrics_table_bootstrap(
    sharpe_vals,
    maxdd_pct_vals,
    final_ret_vals,
    eq0: float,
    actual_final_equity: float,
    actual_final_ret: float,
    actual_maxdd_pct: float,
    actual_maxdd_abs: float,
    actual_sharpe: float,
    maxdd_bca_low: float | None = None,
    maxdd_bca_high: float | None = None,
) -> html.Div:

    if sharpe_vals.size == 0:
        return html.Div("No bootstrap results.", style={"color": "#AAAAAA"})

    def pct(x):
        return float(x) * 100.0

    rows = []

    # Baseline metrics from actual portfolio
    rows.append(("Actual final equity ($)", f"{actual_final_equity:,.0f}"))
    rows.append(("Actual total P&L ($)", f"{(actual_final_equity - eq0):,.0f}"))
    rows.append(
        ("Actual Max DD (% / $)", f"{pct(actual_maxdd_pct):.1f}% / {actual_maxdd_abs:,.0f}")
    )
    rows.append(("Actual Sharpe", f"{actual_sharpe:.2f}"))

    # Distribution stats
    sharpe_med = float(np.median(sharpe_vals))
    sharpe_p05 = float(np.percentile(sharpe_vals, 5))
    sharpe_p95 = float(np.percentile(sharpe_vals, 95))
    sharpe_lt0 = float(np.mean(sharpe_vals < 0.0) * 100.0)

    rows.append(("Sharpe (median)", f"{sharpe_med:.2f}"))
    rows.append(("Sharpe 5% / 95%", f"{sharpe_p05:.2f} / {sharpe_p95:.2f}"))
    rows.append(("Prob(Sharpe < 0)", f"{sharpe_lt0:.1f}%"))

    maxdd_med = float(np.median(maxdd_pct_vals))
    maxdd_p95 = float(np.percentile(maxdd_pct_vals, 95))
    rows.append(("Max DD (median, %)", f"{pct(maxdd_med):.1f}%"))
    rows.append(("Max DD 95% worst, %", f"{pct(maxdd_p95):.1f}%"))
    
    # BCa interval for Max DD (if available)
    if (maxdd_bca_low is not None) and (maxdd_bca_high is not None):
        rows.append(
            (
                "Max DD 95% BCa, %",
                f"{pct(maxdd_bca_low):.1f}% / {pct(maxdd_bca_high):.1f}%",
            )
        )


    fin_med = float(np.median(final_ret_vals))
    fin_p05 = float(np.percentile(final_ret_vals, 5))
    fin_p95 = float(np.percentile(final_ret_vals, 95))
    rows.append(("Final return (median, %)", f"{pct(fin_med):.1f}%"))
    rows.append(
        ("Final return 5% / 95%, %", f"{pct(fin_p05):.1f}% / {pct(fin_p95):.1f}%")
    )

    table_rows = []
    for label, value in rows:
        table_rows.append(
            html.Tr(
                [
                    html.Td(label, style={"padding": "0.15rem 0.4rem"}),
                    html.Td(value, style={"padding": "0.15rem 0.4rem", "textAlign": "right"}),
                ]
            )
        )

    return html.Table(
        table_rows,
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "fontSize": "0.8rem",
        },
    )


def _metrics_table_mc(
    final_equity_vals: np.ndarray,
    maxdd_vals: np.ndarray,
    sharpe_vals: np.ndarray,
    ruin_flags: np.ndarray,
    min_equity_vals: np.ndarray,
    worst_1d_loss_vals: np.ndarray,
    worst_5d_loss_vals: np.ndarray,
    worst_20d_loss_vals: np.ndarray,
    eq0: float,
    actual_final_equity: float,
    actual_maxdd_frac: float,
    actual_sharpe: float,
) -> html.Div:
    if final_equity_vals.size == 0:
        return html.Div("No Monte Carlo results.", style={"color": "#AAAAAA"})

    def pct(x: float) -> float:
        return float(x) * 100.0

    # MC final returns
    if eq0 > 0:
        final_ret_vals = final_equity_vals / eq0 - 1.0
    else:
        final_ret_vals = np.zeros_like(final_equity_vals)

    # --------------------------
    # Build rows by "group"
    # --------------------------
    actual_rows = []       # no shading
    mc_central_rows = []   # blue shading
    dd_rows = []           # yellow shading
    tail_rows = []         # red shading

    # ----- Actual (historical) metrics: no shading -----
    actual_rows.append(("Actual final equity ($)", f"{actual_final_equity:,.0f}"))
    actual_rows.append(("Actual total P&L ($)", f"{(actual_final_equity - eq0):,.0f}"))
    actual_rows.append(("Actual Max DD (%)", f"{pct(actual_maxdd_frac):.1f}%"))
    actual_rows.append(("Actual Sharpe", f"{actual_sharpe:.2f}"))

    # ----- MC central tendency (blue) -----
    fin_med = float(np.median(final_ret_vals))
    fin_p05 = float(np.percentile(final_ret_vals, 5))
    fin_p95 = float(np.percentile(final_ret_vals, 95))
    mc_central_rows.append(("Final return (median, %)", f"{pct(fin_med):.1f}%"))
    mc_central_rows.append(
        (
            "Final return 5% / 95%, %",
            f"{pct(fin_p05):.1f}% / {pct(fin_p95):.1f}%",
        )
    )

    sharpe_med = float(np.median(sharpe_vals))
    sharpe_p05 = float(np.percentile(sharpe_vals, 5))
    sharpe_p95 = float(np.percentile(sharpe_vals, 95))
    mc_central_rows.append(("Sharpe (median)", f"{sharpe_med:.2f}"))
    mc_central_rows.append(
        ("Sharpe 5% / 95%", f"{sharpe_p05:.2f} / {sharpe_p95:.2f}")
    )

    # ----- DD distribution & DD thresholds (yellow) -----
    dd_med = float(np.median(maxdd_vals))
    dd_p95 = float(np.percentile(maxdd_vals, 95))
    dd_rows.append(("Max DD (median, %)", f"{pct(dd_med):.1f}%"))
    dd_rows.append(("Max DD 95% worst, %", f"{pct(dd_p95):.1f}%"))

    for thr in (0.20, 0.30, 0.40):
        prob_dd = float(np.mean(maxdd_vals > thr) * 100.0)
        dd_rows.append((f"Prob(Max DD > {int(thr*100)}%)", f"{prob_dd:.1f}%"))

    # ----- Tail / loss metrics (red) -----

    # Worst loss events in MC (dollars)
    if worst_1d_loss_vals.size > 0:
        worst_1d = float(worst_1d_loss_vals.min())
        tail_rows.append(("Worst 1-day loss in MC ($)", f"{worst_1d:,.0f}"))

    if worst_5d_loss_vals.size > 0:
        worst_5d = float(worst_5d_loss_vals.min())
        tail_rows.append(("Worst 5-day loss in MC ($)", f"{worst_5d:,.0f}"))

    if worst_20d_loss_vals.size > 0:
        worst_20d = float(worst_20d_loss_vals.min())
        tail_rows.append(("Worst 20-day loss in MC ($)", f"{worst_20d:,.0f}"))

    # Worst MIN equity across MC paths
    if min_equity_vals.size > 0:
        worst_min_equity = float(min_equity_vals.min())
        tail_rows.append(("Worst MIN equity in MC ($)", f"{worst_min_equity:,.0f}"))

    # VaR / ES on final return (loss perspective)
    var99 = float(np.percentile(final_ret_vals, 1))  # 1% quantile
    tail = final_ret_vals[final_ret_vals <= var99]
    es99 = float(tail.mean()) if tail.size > 0 else var99

    tail_rows.append(
        (
            "99% VaR (loss, %, $)",
            f"{-pct(var99):.1f}% / {-var99 * eq0:,.0f}",
        )
    )
    tail_rows.append(
        (
            "99% ES (loss, %, $)",
            f"{-pct(es99):.1f}% / {-es99 * eq0:,.0f}",
        )
    )

    prob_sharpe_lt0 = float(np.mean(sharpe_vals < 0.0) * 100.0)
    prob_sharpe_lt_half = float(np.mean(sharpe_vals < actual_sharpe / 2.0) * 100.0)
    tail_rows.append(("Prob(Sharpe < 0)", f"{prob_sharpe_lt0:.1f}%"))
    tail_rows.append(("Prob(Sharpe < ½ actual)", f"{prob_sharpe_lt_half:.1f}%"))

    prob_ruin = float(np.mean(ruin_flags) * 100.0)
    tail_rows.append(("Prob(ruin: equity < threshold)", f"{prob_ruin:.2f}%"))

    # --------------------------
    # Build HTML with shading
    # --------------------------
    # Colours tuned for dark theme; adjust to taste
    # blue_bg = "#23324a"
    # yellow_bg = "#4a4323"
    # red_bg = "#4a2323"
    blue_bg   = "rgba(35, 50, 74, 0.35)"     # adjust alpha as desired
    yellow_bg = "rgba(74, 67, 35, 0.35)"
    red_bg    = "rgba(74, 35, 35, 0.35)"


    table_rows = []

    # Actual rows – no backgroundColor
    for label, value in actual_rows:
        table_rows.append(
            html.Tr(
                [
                    html.Td(label, style={"padding": "0.15rem 0.4rem"}),
                    html.Td(
                        value,
                        style={
                            "padding": "0.15rem 0.4rem",
                            "textAlign": "right",
                        },
                    ),
                ]
            )
        )

    # MC central tendency – blue shading
    for label, value in mc_central_rows:
        table_rows.append(
            html.Tr(
                [
                    html.Td(
                        label,
                        style={
                            "padding": "0.15rem 0.4rem",
                            "backgroundColor": blue_bg,
                        },
                    ),
                    html.Td(
                        value,
                        style={
                            "padding": "0.15rem 0.4rem",
                            "textAlign": "right",
                            "backgroundColor": blue_bg,
                        },
                    ),
                ]
            )
        )

    # DD metrics – yellow shading
    for label, value in dd_rows:
        table_rows.append(
            html.Tr(
                [
                    html.Td(
                        label,
                        style={
                            "padding": "0.15rem 0.4rem",
                            "backgroundColor": yellow_bg,
                        },
                    ),
                    html.Td(
                        value,
                        style={
                            "padding": "0.15rem 0.4rem",
                            "textAlign": "right",
                            "backgroundColor": yellow_bg,
                        },
                    ),
                ]
            )
        )

    # Tail metrics – red shading
    for label, value in tail_rows:
        table_rows.append(
            html.Tr(
                [
                    html.Td(
                        label,
                        style={
                            "padding": "0.15rem 0.4rem",
                            "backgroundColor": red_bg,
                        },
                    ),
                    html.Td(
                        value,
                        style={
                            "padding": "0.15rem 0.4rem",
                            "textAlign": "right",
                            "backgroundColor": red_bg,
                        },
                    ),
                ]
            )
        )

    return html.Table(
        table_rows,
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "fontSize": "0.8rem",
        },
    )


def _window_metrics_from_pnl(pnl_window: np.ndarray, eq0: float) -> tuple[float, float, int]:
    """
    Returns:
      total_pnl ($), max_dd_pct (fraction, e.g. 0.12), dd_duration_days (int)

    DD duration here = days between the peak and the trough that define the maximum drawdown.
    """
    pnl_window = np.asarray(pnl_window, dtype=float)
    if pnl_window.size == 0 or eq0 <= 0:
        return 0.0, 0.0, 0

    eq = eq0 + np.cumsum(pnl_window)

    running_max = np.maximum.accumulate(eq)
    dd = (running_max - eq) / running_max  # fraction
    max_dd = float(dd.max()) if dd.size else 0.0

    # dd duration: peak index -> trough index for the max DD event
    trough_idx = int(np.argmax(dd)) if dd.size else 0
    peak_idx = int(np.argmax(eq[: trough_idx + 1])) if trough_idx > 0 else 0
    dd_dur = int(max(0, trough_idx - peak_idx))

    return float(pnl_window.sum()), max_dd, dd_dur

def _random_start_date_analysis(
    daily_pnl: pd.Series,
    months: int,
    n_periods: int,
    eq0: float,
    no_overlap: bool = False,
    seed: int | None = None,
) -> dict:
    """
    daily_pnl: pd.Series indexed by datetime-like, values are DAILY $ P&L.
    months: window length in calendar months (uses pd.DateOffset).
    n_periods: number of random windows
    eq0: initial equity used to compute DD%
    no_overlap: best-effort to avoid overlapping windows
    """
    s = daily_pnl.dropna()
    s = s.sort_index()
    if s.empty:
        return {"rows": [], "error": "Empty portfolio daily P&L series."}

    # Determine eligible start dates where start+months <= last date
    last_dt = s.index.max()
    offset = pd.DateOffset(months=int(months))

    eligible = []
    for dt in s.index:
        if dt + offset <= last_dt:
            eligible.append(dt)

    if not eligible:
        return {"rows": [], "error": "No eligible start dates for the requested window length."}

    rng = np.random.default_rng(seed)
    rows = []

    used_intervals = []  # list of (start, end) for overlap screening

    attempts = 0
    max_attempts = max(2000, n_periods * 50)

    while len(rows) < n_periods and attempts < max_attempts:
        attempts += 1
        start = eligible[int(rng.integers(0, len(eligible)))]
        end = start + offset

        if no_overlap:
            # reject if overlap with any previously accepted interval (best effort)
            overlap = False
            for (s0, e0) in used_intervals:
                if not (end < s0 or start > e0):
                    overlap = True
                    break
            if overlap:
                continue

        w = s.loc[(s.index >= start) & (s.index <= end)]
        if w.size < 10:
            continue

        total_pnl, max_dd, dd_dur = _window_metrics_from_pnl(w.values, eq0)

        rows.append(
            {
                "start": start,
                "end": w.index.max(),
                "days": int(w.size),
                "total_pnl": float(total_pnl),
                "ret_pct": float(total_pnl / eq0) if eq0 > 0 else 0.0,
                "max_dd": float(max_dd),  # fraction
                "dd_dur": int(dd_dur),
            }
        )

        if no_overlap:
            used_intervals.append((start, w.index.max()))

    if len(rows) < n_periods:
        return {"rows": rows, "error": f"Only generated {len(rows)} windows (requested {n_periods})."}

    return {"rows": rows, "error": None}


def _randstart_figure(rows: list[dict]) -> go.Figure:
    fig = go.Figure()

    # Empty state
    if not rows:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            title={
                "text": "Random start-date analysis (contiguous windows)",
                "x": 0.01,
                "xanchor": "left",
            },
            xaxis={"title": "Random window index"},
            yaxis={"title": "Total P&L ($)"},
        )
        return fig

    # X-axis labels and series
    x_labels = [f"P{i+1}" for i in range(len(rows))]
    pnl = np.array([r["total_pnl"] for r in rows], dtype=float)
    maxdd_pct = np.array([r["max_dd"] * 100.0 for r in rows], dtype=float)

    hover = [
        f"Start: {r['start'].date()}<br>"
        f"End: {r['end'].date()}<br>"
        f"Days: {r['days']}<br>"
        f"Total P&L: ${r['total_pnl']:,.0f}<br>"
        f"Return: {r['ret_pct']*100:.1f}%<br>"
        f"Max DD: {r['max_dd']*100:.1f}%<br>"
        f"DD duration: {r['dd_dur']} days"
        for r in rows
    ]

    # ---------------------------
    # Main series
    # ---------------------------

    # Total P&L on left axis
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=pnl,
            mode="lines+markers",
            name="Total P&L ($)",
            hovertext=hover,
            hoverinfo="text",
            legendgroup="pnl",
            showlegend=True,
        )
    )

    # Max DD on right axis
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=maxdd_pct,
            mode="lines+markers",
            name="Max drawdown (%)",
            yaxis="y2",
            hovertext=hover,
            hoverinfo="text",
            legendgroup="dd",
            showlegend=True,
        )
    )

    # ---------------------------
    # Percentile lines (toggle with their group)
    # ---------------------------

    # P&L percentiles
    pnl_p25 = float(np.percentile(pnl, 25))
    pnl_p50 = float(np.percentile(pnl, 50))
    pnl_p75 = float(np.percentile(pnl, 75))

    fig.add_trace(
        go.Scatter(
            x=[x_labels[0], x_labels[-1]],
            y=[pnl_p25, pnl_p25],
            mode="lines",
            name="P&L 25th",
            line=dict(color="rgba(135,206,250,0.6)", width=1, dash="dot"),
            legendgroup="pnl",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x_labels[0], x_labels[-1]],
            y=[pnl_p50, pnl_p50],
            mode="lines",
            name="P&L median",
            line=dict(color="rgba(135,206,250,0.9)", width=2, dash="dash"),
            legendgroup="pnl",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x_labels[0], x_labels[-1]],
            y=[pnl_p75, pnl_p75],
            mode="lines",
            name="P&L 75th",
            line=dict(color="rgba(135,206,250,0.6)", width=1, dash="dot"),
            legendgroup="pnl",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Max DD percentiles
    dd_p25 = float(np.percentile(maxdd_pct, 25))
    dd_p50 = float(np.percentile(maxdd_pct, 50))
    dd_p75 = float(np.percentile(maxdd_pct, 75))

    fig.add_trace(
        go.Scatter(
            x=[x_labels[0], x_labels[-1]],
            y=[dd_p25, dd_p25],
            mode="lines",
            name="DD 25th",
            line=dict(color="rgba(255,165,0,0.6)", width=1, dash="dot"),
            yaxis="y2",
            legendgroup="dd",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x_labels[0], x_labels[-1]],
            y=[dd_p50, dd_p50],
            mode="lines",
            name="DD median",
            line=dict(color="rgba(255,165,0,0.9)", width=2, dash="dash"),
            yaxis="y2",
            legendgroup="dd",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x_labels[0], x_labels[-1]],
            y=[dd_p75, dd_p75],
            mode="lines",
            name="DD 75th",
            line=dict(color="rgba(255,165,0,0.6)", width=1, dash="dot"),
            yaxis="y2",
            legendgroup="dd",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        title={
            "text": "Random start-date analysis (contiguous windows)",
            "x": 0.01,
            "xanchor": "left",
        },
        xaxis={"title": "Random window index"},
        yaxis={"title": "Total P&L ($)"},
        yaxis2={
            "title": "Max drawdown (%)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        legend={
            "orientation": "h",
            "y": -0.2,
            "groupclick": "togglegroup",
        },
        margin=dict(l=60, r=60, t=50, b=60),
    )

    return fig




def _randstart_dist_figure(rows: list[dict]) -> go.Figure:
    """
    Distribution view for random windows:
    - Left: Total P&L histogram + violin
    - Right: Max DD (%) histogram + violin
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Total P&L distribution", "Max DD distribution (%)"),
        horizontal_spacing=0.18,
    )

    if not rows:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
        )
        return fig

    pnl = np.array([r["total_pnl"] for r in rows], dtype=float)
    maxdd_pct = np.array([r["max_dd"] * 100.0 for r in rows], dtype=float)

    # Total P&L
    fig.add_trace(
        go.Histogram(x=pnl, nbinsx=20, name="P&L hist", opacity=0.75),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Violin(
            x=pnl,
            name="P&L violin",
            points=False,
            box_visible=True,
            meanline_visible=True,
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    # Max DD %
    fig.add_trace(
        go.Histogram(x=maxdd_pct, nbinsx=20, name="DD hist", opacity=0.75),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Violin(
            x=maxdd_pct,
            name="DD violin",
            points=False,
            box_visible=True,
            meanline_visible=True,
            opacity=0.7,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Total P&L ($)", row=1, col=1)
    fig.update_xaxes(title_text="Max DD (%)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        legend=dict(orientation="h", y=-0.4),
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def _randstart_metrics(rows: list[dict]) -> html.Table | html.Div:
    """
    Small dispersion table for random-start windows.
    """
    if not rows:
        return html.Div("No random windows generated.", style={"color": "#AAAAAA"})

    pnl = np.array([r["total_pnl"] for r in rows], dtype=float)
    ret_pct = np.array([r["ret_pct"] * 100.0 for r in rows], dtype=float)
    maxdd_pct = np.array([r["max_dd"] * 100.0 for r in rows], dtype=float)

    def iqr(x: np.ndarray) -> float:
        return float(np.percentile(x, 75) - np.percentile(x, 25))

    rows_out: list[tuple[str, str]] = []
    rows_out.append(("N windows", f"{len(pnl)}"))

    # P&L block
    rows_out.append(("\u2014 Total P&L ($) \u2014", ""))
    pnl_med = float(np.median(pnl))
    pnl_p25 = float(np.percentile(pnl, 25))
    pnl_p75 = float(np.percentile(pnl, 75))
    rows_out.append(("Median", f"{pnl_med:,.0f}"))
    rows_out.append(("P25 / P75", f"{pnl_p25:,.0f} / {pnl_p75:,.0f}"))
    rows_out.append(("Min / Max", f"{pnl.min():,.0f} / {pnl.max():,.0f}"))
    if pnl_med != 0:
        rows_out.append(
            ("IQR / |median|", f"{iqr(pnl) / abs(pnl_med):.2f}")
        )

    # Return %
    rows_out.append(("\u2014 Return (%) \u2014", ""))
    ret_med = float(np.median(ret_pct))
    ret_p25 = float(np.percentile(ret_pct, 25))
    ret_p75 = float(np.percentile(ret_pct, 75))
    rows_out.append(("Median", f"{ret_med:.1f}%"))
    rows_out.append(("P25 / P75", f"{ret_p25:.1f}% / {ret_p75:.1f}%"))

    # Max DD %
    rows_out.append(("\u2014 Max DD (%) \u2014", ""))
    dd_med = float(np.median(maxdd_pct))
    dd_p25 = float(np.percentile(maxdd_pct, 25))
    dd_p75 = float(np.percentile(maxdd_pct, 75))
    rows_out.append(("Median", f"{dd_med:.1f}%"))
    rows_out.append(("P25 / P75", f"{dd_p25:.1f}% / {dd_p75:.1f}%"))
    rows_out.append(("Min / Max", f"{maxdd_pct.min():.1f}% / {maxdd_pct.max():.1f}%"))

    table_rows = []
    for label, value in rows_out:
        is_header = value == ""
        style_label = {"padding": "0.15rem 0.4rem"}
        style_val = {
            "padding": "0.15rem 0.4rem",
            "textAlign": "right",
        }
        if is_header:
            style_label.update({"fontWeight": "bold", "paddingTop": "0.35rem"})
        table_rows.append(
            html.Tr(
                [
                    html.Td(label, style=style_label),
                    html.Td(value, style=style_val),
                ]
            )
        )

    return html.Table(
        table_rows,
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "fontSize": "0.8rem",
        },
    )



# ---------------------------------------------------------------------------
# Callbacks – Bootstrap
# ---------------------------------------------------------------------------


@callback(
    Output("p2-robust-boot-sharpe-hist", "figure"),
    Output("p2-robust-boot-maxdd-hist", "figure"),
    Output("p2-robust-boot-maxblock-hist", "figure"),
    Output("p2-robust-boot-final-ecdf", "figure"),
    Output("p2-robust-boot-maxdd-ecdf", "figure"),
    Output("p2-robust-boot-metrics-table", "children"),
    Output("p2-robust-boot-status", "children"),
    Input("p2-robust-boot-run-btn", "n_clicks"),
    State("p1-active-list-store", "data"),
    State("p2-weights-store", "data"),
    State("p2-robust-initial-equity-input", "value"),
    State("p2-robust-boot-n-sim", "value"),
    State("p2-robust-boot-block-len", "value"),
)
def run_portfolio_bootstrap(
    n_clicks,
    active_store,
    weights_store,
    initial_equity_value,
    n_sim_value,
    block_len_value,
):
    if not n_clicks:
        msg = "Click 'Run Bootstrap' to generate simulations."
        empty = _empty_robustness_figure(msg)
        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            html.Div(msg, style={"color": "#AAAAAA"}),
            "",
        )

    pnl, eq0 = _extract_portfolio_pnl(active_store, weights_store, initial_equity_value)
    if pnl is None:
        msg = "Unable to build portfolio daily P&L. Check selected strategies and date overlap."
        empty = _empty_robustness_figure(msg)
        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            html.Div(msg, style={"color": "#AAAAAA"}),
            msg,
        )

    try:
        n_sim = int(n_sim_value or 0)
        block_len = int(block_len_value or 0)
    except Exception:
        msg = "Invalid bootstrap parameters."
        empty = _empty_robustness_figure(msg)
        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            html.Div(msg, style={"color": "#AAAAAA"}),
            msg,
        )

    pnl_arr = pnl.values.astype(float)

    # Actual metrics from additive P&L
    actual_sharpe = _sharpe_from_pnl(pnl_arr, eq0)
    actual_maxdd_abs, actual_maxdd_pct = _max_dd_from_pnl(pnl_arr, eq0)
    actual_final_equity = float(eq0 + pnl_arr.sum())
    actual_final_ret = actual_final_equity / eq0 - 1.0 if eq0 > 0 else 0.0
    actual_maxblock_pct = _max_block_loss_from_pnl(pnl_arr, block_len, eq0)

    (
        sharpe_vals,
        maxdd_pct_vals,
        final_ret_vals,
        worst_block_pct_vals,
        final_equity_vals,
    ) = _bootstrap_distribution_from_pnl(pnl, n_sim, block_len, eq0)
    
    # BCa 95% interval for Max DD (fraction units, e.g. 0.2 = 20%)
    try:
        maxdd_bca_low, maxdd_bca_high = _bca_interval_maxdd(
            pnl_arr, eq0, maxdd_pct_vals
        )
    except Exception:
        maxdd_bca_low, maxdd_bca_high = None, None


    # Histograms with Actual + 5/95 vlines
    sharpe_fig = _hist_figure(
        sharpe_vals,
        "Bootstrap – Sharpe ratio distribution",
        "Sharpe",
        orig_value=actual_sharpe,
        orig_hover=f"Actual Sharpe: {actual_sharpe:.2f}",
        p05=float(np.percentile(sharpe_vals, 5)),
        p95=float(np.percentile(sharpe_vals, 95)),
    )
    maxdd_percent = maxdd_pct_vals * 100.0
    bca_low_pct = maxdd_bca_low * 100.0 if maxdd_bca_low is not None else None
    bca_high_pct = maxdd_bca_high * 100.0 if maxdd_bca_high is not None else None

    maxdd_fig = _hist_figure(
        maxdd_percent,
        "Bootstrap – Max drawdown distribution",
        "Max drawdown (%)",
        orig_value=actual_maxdd_pct * 100.0,
        orig_hover=f"Actual Max DD: {actual_maxdd_pct * 100.0:.1f}%",
        p05=float(np.percentile(maxdd_percent, 5)),
        p95=float(np.percentile(maxdd_percent, 95)),
        bca_low=bca_low_pct,
        bca_high=bca_high_pct,
    )

    maxblock_percent = worst_block_pct_vals * 100.0
    maxblock_fig = _hist_figure(
        maxblock_percent,
        f"Bootstrap – Worst {block_len}-day block return",
        "Worst block return (%)",
        orig_value=actual_maxblock_pct * 100.0,
        orig_hover=f"Actual worst {block_len}-day block: {actual_maxblock_pct * 100.0:.1f}%",
        p05=float(np.percentile(maxblock_percent, 5)),
        p95=float(np.percentile(maxblock_percent, 95)),
    )

    # ECDFs
    final_ecdf_fig = _ecdf_figure(
        final_equity_vals,
        actual_final_equity,
        "Bootstrap – ECDF of final portfolio equity",
        "Final equity ($)",
        orig_hover=f"Actual final equity: ${actual_final_equity:,.0f}",
    )
    maxdd_ecdf_fig = _ecdf_figure(
        maxdd_percent,
        actual_maxdd_pct * 100.0,
        "Bootstrap – ECDF of Max drawdown",
        "Max drawdown (%)",
        orig_hover=f"Actual Max DD: {actual_maxdd_pct * 100.0:.1f}%",
    )

    metrics = _metrics_table_bootstrap(
        sharpe_vals,
        maxdd_pct_vals,
        final_ret_vals,
        eq0,
        actual_final_equity,
        actual_final_ret,
        actual_maxdd_pct,
        actual_maxdd_abs,
        actual_sharpe,
        maxdd_bca_low=maxdd_bca_low,
        maxdd_bca_high=maxdd_bca_high,
    )


    status_msg = f"Bootstrap completed: {n_sim} runs, block length = {block_len} days."

    return (
        sharpe_fig,
        maxdd_fig,
        maxblock_fig,
        final_ecdf_fig,
        maxdd_ecdf_fig,
        metrics,
        status_msg,
    )


# ---------------------------------------------------------------------------
# Callbacks – Monte Carlo
# ---------------------------------------------------------------------------


@callback(
    Output("p2-robust-mc-final-equity-hist", "figure"),
    Output("p2-robust-mc-maxdd-hist", "figure"),
    Output("p2-robust-mc-fan-chart", "figure"),
    Output("p2-robust-mc-final-ecdf", "figure"),
    Output("p2-robust-mc-ruin-bar", "figure"),
    Output("p2-robust-mc-metrics-table", "children"),
    Output("p2-robust-mc-status", "children"),
    Input("p2-robust-mc-run-btn", "n_clicks"),
    State("p1-active-list-store", "data"),
    State("p2-weights-store", "data"),
    State("p2-robust-initial-equity-input", "value"),
    State("p2-robust-mc-n-sim", "value"),
    State("p2-robust-mc-horizon-days", "value"),
    State("p2-robust-ruin-threshold", "value"),
)


def run_portfolio_mc(
    n_clicks,
    active_store,
    weights_store,
    initial_equity_value,
    n_sim_value,
    horizon_days_value,
    ruin_threshold_value,
):

    if not n_clicks:
        msg = "Click 'Run Monte Carlo' to generate simulations."
        empty = _empty_robustness_figure(msg)
        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            html.Div(msg, style={"color": "#AAAAAA"}),
            "",
        )

    pnl, eq0 = _extract_portfolio_pnl(active_store, weights_store, initial_equity_value)
    if pnl is None:
        msg = "Unable to build portfolio daily P&L. Check selected strategies and date overlap."
        empty = _empty_robustness_figure(msg)
        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            html.Div(msg, style={"color": "#AAAAAA"}),
            msg,
        )

    try:
        n_sim = int(n_sim_value or 0)
        horizon_days = int(horizon_days_value or 0)
    except Exception:
        msg = "Invalid Monte Carlo parameters."
        empty = _empty_robustness_figure(msg)
        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            html.Div(msg, style={"color": "#AAAAAA"}),
            msg,
        )
    
    # Parse ruin threshold ($)
    try:
        ruin_threshold = float(
            ruin_threshold_value if ruin_threshold_value is not None else 0.0
        )
    except Exception:
        ruin_threshold = 0.0


    pnl_arr = pnl.values.astype(float)

    # Actual baseline from additive P&L over the last horizon_days (if specified)
    if horizon_days > 0 and horizon_days <= len(pnl_arr):
        pnl_window = pnl_arr[-horizon_days:]
    else:
        pnl_window = pnl_arr
        horizon_days = len(pnl_window)

    actual_equity_path = eq0 + np.cumsum(pnl_window)
    actual_final_equity = float(actual_equity_path[-1])
    #actual_final_ret = actual_final_equity / eq0 - 1.0 if eq0 > 0 else 0.0
    actual_min_equity = float(actual_equity_path.min())

    # Max DD and Sharpe on this window
    peak = np.maximum.accumulate(actual_equity_path)
    dd_frac = 1.0 - actual_equity_path / peak
    actual_maxdd_frac = float(dd_frac.max())

    if eq0 > 0:
        returns_hist = pnl_window / eq0
    else:
        returns_hist = np.zeros_like(pnl_window)

    mu_hist = float(returns_hist.mean())
    sigma_hist = float(returns_hist.std(ddof=1))
    if sigma_hist > 0:
        actual_sharpe = mu_hist / sigma_hist * np.sqrt(252.0)
    else:
        actual_sharpe = 0.0

    # Empirical MC on daily P&L
    pnl_series = pd.Series(pnl_window)
    (
        final_equity_vals,
        maxdd_vals,
        sharpe_vals,
        ruin_flags,
        stored_paths,
        min_equity_vals,
        worst_1d_loss_vals,
        worst_5d_loss_vals,
        worst_20d_loss_vals,
        n_days_used,
    ) = _mc_distribution_empirical(
        pnl_series, n_sim, horizon_days, eq0, ruin_threshold
    )





    if final_equity_vals.size == 0:
        msg = "Monte Carlo produced no valid paths."
        empty = _empty_robustness_figure(msg)
        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            html.Div(msg, style={"color": "#AAAAAA"}),
            msg,
        )

    # Histograms (final equity & Max DD)
    final_fig = _hist_figure(
        final_equity_vals,
        "Monte Carlo – Final equity distribution",
        "Final equity ($)",
        orig_value=actual_final_equity,
        orig_hover=f"Actual final equity: ${actual_final_equity:,.0f}",
        p05=float(np.percentile(final_equity_vals, 5)),
        p95=float(np.percentile(final_equity_vals, 95)),
    )

    maxdd_percent = maxdd_vals * 100.0
    maxdd_fig = _hist_figure(
        maxdd_percent,
        "Monte Carlo – Max drawdown distribution",
        "Max drawdown (%)",
        orig_value=actual_maxdd_frac * 100.0,
        orig_hover=f"Actual Max DD: {actual_maxdd_frac * 100.0:.1f}%",
        p05=float(np.percentile(maxdd_percent, 5)),
        p95=float(np.percentile(maxdd_percent, 95)),
    )

    # Fan chart
    fan_fig = _empty_robustness_figure("No MC paths to display.")
    if stored_paths and n_days_used > 0:
        fan_fig = go.Figure()
        x_axis = np.arange(1, n_days_used + 1)

        # stack stored paths for percentiles
        paths_matrix = np.vstack(stored_paths)  # shape: (k, n_days)
        # faint individual paths
        for path in stored_paths:
            fan_fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=path,
                    mode="lines",
                    line=dict(width=1),
                    opacity=0.1,
                    showlegend=False,
                )
            )

        # percentiles
        median_path = np.percentile(paths_matrix, 50, axis=0)
        p05_path = np.percentile(paths_matrix, 5, axis=0)
        p95_path = np.percentile(paths_matrix, 95, axis=0)

        fan_fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=median_path,
                mode="lines",
                name="Median path",
                line=dict(width=2),
            )
        )
        fan_fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=p05_path,
                mode="lines",
                name="5th percentile",
                line=dict(width=1, dash="dot"),
            )
        )
        fan_fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=p95_path,
                mode="lines",
                name="95th percentile",
                line=dict(width=1, dash="dot"),
            )
        )

        # actual historical equity path over the same horizon
        fan_fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=actual_equity_path[-n_days_used:],
                mode="lines",
                name="Actual history",
                line=dict(width=2),
            )
        )

        fan_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            margin=dict(l=40, r=20, t=40, b=50),
            title="Monte Carlo – Fan chart (equity paths)",
            xaxis_title="Days",
            yaxis_title="Equity ($)",
        )

    # ECDF of MINIMUM equity (for ruin analysis)
    actual_min_equity = float(actual_equity_path.min())

    final_ecdf_fig = _ecdf_figure(
        min_equity_vals,
        actual_min_equity,
        "Monte Carlo – ECDF of MINIMUM portfolio equity",
        "Minimum equity ($)",
        orig_hover=f"Actual minimum equity: ${actual_min_equity:,.0f}",
    )


    # probability of ruin (based on path-level ruin flags)
    prob_ruin = float(np.mean(ruin_flags))

    # mark chosen ruin threshold on ECDF of final equity
    final_ecdf_fig.add_vline(
        x=ruin_threshold,
        line_color="orange",
        line_width=1,
        line_dash="dash",
        annotation_text=f"Ruin threshold, Prob={prob_ruin*100:.2f}%",
        annotation_position="bottom left",
    )


    # Probability-of-ruin bar chart
    ruin_bar_fig = go.Figure()
    prob_ruin_pct = prob_ruin * 100.0
    # ensure a tiny visible bar even if probability is exactly zero
    display_y = max(prob_ruin_pct, 0.01)
    
    ruin_bar_fig.add_trace(
        go.Bar(
            x=["Ruin probability"],
            y=[display_y],
            text=[f"{prob_ruin_pct:.2f}%"],
            textposition="outside",
            showlegend=False,
        )
    )
    ruin_bar_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin=dict(l=40, r=20, t=40, b=50),
        title="Probability of ruin (final equity < 0)",
    )
    ruin_bar_fig.update_yaxes(
        title_text="Probability (%)",
        range=[0, max(1.0, display_y * 1.2)],
    )


    # Metrics table
    metrics = _metrics_table_mc(
        final_equity_vals,
        maxdd_vals,
        sharpe_vals,
        ruin_flags,
        min_equity_vals,
        worst_1d_loss_vals,
        worst_5d_loss_vals,
        worst_20d_loss_vals,
        eq0,
        actual_final_equity,
        actual_maxdd_frac,
        actual_sharpe,
    )



    status_msg = (
        f"Monte Carlo (empirical) completed: {n_sim} paths, "
        f"horizon = {horizon_days} days."
    )

    return (
        final_fig,
        maxdd_fig,
        fan_fig,
        final_ecdf_fig,
        ruin_bar_fig,
        metrics,
        status_msg,
    )


@callback(
    Output("p2-randstart-fig", "figure"),
    Output("p2-randstart-dist-fig", "figure"),
    Output("p2-randstart-metrics", "children"),
    Output("p2-randstart-status", "children"),
    Input("p2-randstart-run-btn", "n_clicks"),
    State("p2-randstart-n-periods", "value"),
    State("p2-randstart-months", "value"),
    State("p2-randstart-no-overlap", "value"),
    State("p1-active-list-store", "data"),
    State("p2-weights-store", "data"),
    State("p2-robust-initial-equity-input", "value"),
)
def run_random_start_date_analysis(
    n_clicks,
    n_periods,
    months,
    no_overlap_val,
    active_store,
    weights_store,
    initial_equity_value,
):
    # Initial blank state – consistent dark figures
    if not n_clicks:
        msg = "Click 'Run random start-date analysis' to generate results."
        empty_main = _empty_robustness_figure(msg)
        empty_dist = _empty_robustness_figure("Distribution will appear here after running the test.")
        metrics = html.Div("No results yet.", style={"color": "#AAAAAA"})
        return empty_main, empty_dist, metrics, msg

    # Defensive defaults
    n_periods = int(n_periods or 50)
    months = int(months or 6)
    no_overlap = bool(no_overlap_val) and ("no_overlap" in no_overlap_val)

    # Build portfolio daily P&L (same as Bootstrap/MC)
    pnl, eq0 = _extract_portfolio_pnl(active_store, weights_store, initial_equity_value)
    if pnl is None:
        msg = "Unable to build portfolio daily P&L. Check selected strategies and date overlap."
        empty = _empty_robustness_figure(msg)
        metrics = html.Div(msg, style={"color": "#AAAAAA"})
        return empty, empty, metrics, msg

    out = _random_start_date_analysis(
        daily_pnl=pnl,
        months=months,
        n_periods=n_periods,
        eq0=eq0,
        no_overlap=no_overlap,
        seed=None,
    )

    rows = out.get("rows") or []
    if len(rows) == 0:
        msg = out.get("error") or "No windows produced. Try reducing window length or disabling no-overlap."
        empty = _empty_robustness_figure(msg)
        metrics = html.Div(msg, style={"color": "#AAAAAA"})
        return empty, empty, metrics, msg

    fig_main = _randstart_figure(rows)
    fig_dist = _randstart_dist_figure(rows)
    metrics = _randstart_metrics(rows)
    msg = out.get("error") or f"Generated {len(rows)} windows of {months} months."

    return fig_main, fig_dist, metrics, msg
