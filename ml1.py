# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 12:15:55 2025

@author: mauro

Portfolio26 - Phase 3 ML
ml1.py

Single-run Walk-Forward Analysis (FWA) using LightGBM classifier.

Key rules:
- Training set (IS) is filtered by close_dt (information availability).
- Prediction candidates (OoS) are filtered by open_dt (decision time).
- Equity curve is stitched by close_dt (realistic DD/path).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# -------------------------
# Optional dependency
# -------------------------
try:
    from lightgbm import LGBMClassifier
except Exception as e:
    LGBMClassifier = None
    _LGBM_IMPORT_ERROR = e


# -------------------------
# Parameters
# -------------------------
@dataclass
class RunParams:
    dataset_csv_path: str

    # date bounds (can be overridden)
    start_date: Optional[str] = None   # "YYYY-MM-DD" or None = use data min
    end_date: Optional[str] = None     # "YYYY-MM-DD" or None = use data max

    # window config (calendar months)
    is_months: int = 2
    oos_months: int = 1

    anchored_type: str = "U"           # "U" (Unanchored) or "A" (Anchored)
    step_months: int = 1               # default 1 month step

    # selection rule
    top_k_per_day: int = 3

    # model config (v1 defaults)
    lgbm_params: Dict[str, Any] = None

    # output verbosity
    verbose_cycles: bool = True
    
    # equity / normalization
    initial_equity: float = 100000.0



FEATURE_COLS = [
    "dow",
    "open_minute",
    "opening_price",
    "premium",
    "margin_req",
    "opening_vix",
    "gap",
]
TARGET_COL = "label"   # binary target in dataset
PNL_COL = "pnl"
PNLR_COL = "pnl_R"
PREMIUM_COL = "premium"



# -------------------------
# Utilities: date windows
# -------------------------
def _to_ts(d: str) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    # end of month for a given date
    return (pd.Timestamp(year=ts.year, month=ts.month, day=1) + pd.offsets.MonthEnd(0)).normalize()


def _add_months_month_start(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    # move by n months and normalize to first of month
    out = ts + pd.DateOffset(months=n)
    return _month_start(out)


def _compute_cycles(params: RunParams, data_start: pd.Timestamp, data_end: pd.Timestamp) -> pd.DataFrame:
    """
    Build a table of cycles with IS/OoS boundaries.
    All boundaries are calendar-month aligned.
    """
    start = _to_ts(params.start_date) if params.start_date else _month_start(data_start)
    end = _to_ts(params.end_date) if params.end_date else _month_end(data_end)

    start = _month_start(start)
    end = _month_end(end)

    cycles = []
    is_from_0 = start

    i = 0
    while True:
        step_i = i * params.step_months

        if params.anchored_type.upper() == "A":
            # Anchored: IS_from fixed, IS_to expands forward by step
            is_from = is_from_0
            is_to = _month_end(is_from_0 + pd.DateOffset(months=(params.is_months - 1) + step_i))
        else:
            # Unanchored: sliding fixed-length IS window
            is_from = _add_months_month_start(is_from_0, step_i)
            is_to = _month_end(is_from + pd.DateOffset(months=params.is_months - 1))

        oos_from = _month_start(is_to + pd.Timedelta(days=1))
        oos_to = _month_end(oos_from + pd.DateOffset(months=params.oos_months - 1))


        # stop when OoS exceeds end
        if oos_from > end:
            break
        if oos_to > end:
            oos_to = end

        cycles.append(
            {
                "cycle": len(cycles) + 1,
                "IS_from": is_from,
                "IS_to": is_to,
                "OoS_from": oos_from,
                "OoS_to": oos_to,
            }
        )

        # next cycle
        i += 1
        # safety
        if len(cycles) > 500:
            raise RuntimeError("Too many cycles generated; check parameters.")

    return pd.DataFrame(cycles)


# -------------------------
# Metrics
# -------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())  # negative number


def _print_cycles_table(cycles_df: pd.DataFrame) -> None:
    print("\nCYCLES (calendar months):")
    print(cycles_df.to_string(index=False))


def _pcr_from_pnl_and_premium(pnl: pd.Series, premium: pd.Series) -> float:
    """
    Premium Capture Rate (PCR) = sum(P&L) / sum(abs(premium)).
    Returned as a fraction (e.g. 0.322 = 32.2%).
    """
    if pnl is None or pnl.empty or premium is None or premium.empty:
        return np.nan

    total_pnl = float(pnl.sum())
    total_abs_prem = float(premium.abs().sum())

    if total_abs_prem == 0.0:
        return np.nan

    return total_pnl / total_abs_prem



def _annualized_sharpe(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    if daily_returns is None or len(daily_returns) < 2:
        return np.nan
    mu = float(daily_returns.mean())
    sd = float(daily_returns.std(ddof=1))
    if sd == 0:
        return np.nan
    return float((mu / sd) * np.sqrt(periods_per_year))


def _build_realized(trades: pd.DataFrame, pnl_col: str, pnlr_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    trades must contain close_dt, pnl_col, pnlr_col.
    Returns (daily_realized, monthly_realized).
    """
    if trades is None or trades.empty:
        daily = pd.DataFrame(columns=["close_date", pnl_col, pnlr_col])
        monthly = pd.DataFrame(columns=["close_month", pnl_col, pnlr_col])
        return daily, monthly

    t = trades.copy()
    t["close_date"] = t["close_dt"].dt.date
    t["close_month"] = t["close_dt"].dt.to_period("M").astype(str)

    daily = (
        t.groupby("close_date", as_index=False)[[pnl_col, pnlr_col]]
         .sum()
         .sort_values("close_date")
         .reset_index(drop=True)
    )
    monthly = (
        t.groupby("close_month", as_index=False)[[pnl_col, pnlr_col]]
         .sum()
         .sort_values("close_month")
         .reset_index(drop=True)
    )
    return daily, monthly


def _compute_metrics(trades: pd.DataFrame, initial_equity: float, pnl_col: str = PNL_COL, pnlr_col: str = PNLR_COL) -> Dict[str, Any]:
    """
    Full metric set for a trade list, realized by close_dt (daily & monthly).
    Comparable between baseline and ML-selected.
    """
    if trades is None or trades.empty:
        return {
            "trades": 0,
            "total_pnl": 0.0,
            "total_pnlR": 0.0,
            "return_pct": 0.0,
            "pcr": np.nan,
            "max_dd_$": 0.0,
            "max_dd_%": 0.0,
            "sharpe_daily": np.nan,
            "win_month_pct": np.nan,
            "avg_month_pnl": np.nan,
            "median_month_pnl": np.nan,
            "best_month_pnl": np.nan,
            "worst_month_pnl": np.nan,
        }

    # realized series
    daily, monthly = _build_realized(trades, pnl_col=pnl_col, pnlr_col=pnlr_col)

    # equity + drawdown in $
    eq = initial_equity + daily[pnl_col].cumsum().to_numpy()
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([initial_equity])
    dd = eq - peak
    max_dd_dollar = float(dd.min()) if len(dd) else 0.0

    # Max DD % should correspond to the SAME point-in-time as max_dd_dollar:
    # max_dd_% = max_dd_$ / peak_equity_at_time_of_max_dd_$
    if len(dd):
        # dd may be numpy array or pandas Series. Use argmin for robustness.
        pos = int(np.argmin(dd))
        peak_at_dd = float(peak[pos] if isinstance(peak, np.ndarray) else peak.iloc[pos])
        dd_at = float(dd[pos] if isinstance(dd, np.ndarray) else dd.iloc[pos])
        max_dd_pct = (dd_at / peak_at_dd) if peak_at_dd != 0.0 else 0.0
    else:
        max_dd_pct = 0.0


    # daily returns for Sharpe (simple, equity-normalized)
    daily_ret = daily[pnl_col] / float(initial_equity)
    sharpe = _annualized_sharpe(daily_ret)

    # monthly stats (consistency)
    if monthly.empty:
        win_month_pct = np.nan
        avg_month = np.nan
        med_month = np.nan
        best_month = np.nan
        worst_month = np.nan
    else:
        m = monthly[pnl_col]
        win_month_pct = float((m > 0).mean())
        avg_month = float(m.mean())
        med_month = float(m.median())
        best_month = float(m.max())
        worst_month = float(m.min())

    total_pnl = float(trades[pnl_col].sum())
    total_pnlR = float(trades[pnlr_col].sum())
    ret_pct = float(total_pnl / float(initial_equity))

    return {
        "trades": int(len(trades)),
        "total_pnl": total_pnl,
        "total_pnlR": total_pnlR,
        "return_pct": ret_pct,
        "pcr": _pcr_from_pnl_and_premium(trades[pnl_col], trades[PREMIUM_COL]),
        "max_dd_$": max_dd_dollar,
        "max_dd_%": max_dd_pct,
        "sharpe_daily": sharpe,
        "win_month_pct": win_month_pct,
        "avg_month_pnl": avg_month,
        "median_month_pnl": med_month,
        "best_month_pnl": best_month,
        "worst_month_pnl": worst_month,
    }


# -------------------------
# Core FWA run
# -------------------------
def run_fwa_single(params: RunParams) -> Dict[str, Any]:
    if LGBMClassifier is None:
        raise ImportError(f"LightGBM is not available: {_LGBM_IMPORT_ERROR}")

    # defaults
    if params.lgbm_params is None:
        params.lgbm_params = dict(
            n_estimators=245,
            learning_rate=0.02,
            num_leaves=25,
            min_child_samples=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=354,
            n_jobs=-1,
            verbosity=-1,
        )

    # Load dataset
    df = pd.read_csv(params.dataset_csv_path)
    df["open_dt"] = pd.to_datetime(df["open_dt"], errors="coerce")
    df["close_dt"] = pd.to_datetime(df["close_dt"], errors="coerce")
    
    # --- DATE-ONLY columns (used for cycle slicing) ---
    df["open_date"] = pd.to_datetime(df["open_date"], errors="coerce").dt.date
    df["close_date"] = pd.to_datetime(df["close_date"], errors="coerce").dt.date


    # basic validation
    needed = ["open_dt", "close_dt", "strategy_uid"] + FEATURE_COLS + [TARGET_COL, PNL_COL, PNLR_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
        
    # Prevent overlapping OoS windows (double counting trades)
    if params.step_months < params.oos_months:
        raise ValueError(
            f"Invalid parameters: STEP ({params.step_months}) must be >= OoS months ({params.oos_months}). "
            "Otherwise OoS windows overlap and trades are double-counted."
        )


    # Do NOT drop rows because of missing features (LGBM can handle NaN features).
    # Only enforce the structural fields needed for slicing and realized P&L.
    df = df.dropna(
    subset=["open_date", "close_date", TARGET_COL, PNLR_COL, PNL_COL]
    )


    df = df.sort_values("open_dt").reset_index(drop=True)

    data_start = pd.Timestamp(df["open_date"].min())
    data_end = pd.Timestamp(df["open_date"].max())


    cycles_df = _compute_cycles(params, data_start, data_end)
    _print_cycles_table(cycles_df)

    print("\nRUN PARAMS:")
    print(f" dataset: {params.dataset_csv_path}")
    print(f" anchored_type: {params.anchored_type}  IS={params.is_months}m  OoS={params.oos_months}m  step={params.step_months}m")
    print(f" top_k_per_day: {params.top_k_per_day}")
    print(f" features: {FEATURE_COLS}")
    print(f" target: {TARGET_COL} (label = pnl_R > 0)")

    # Accumulators
    oos_all_rows = []       # baseline: ALL OoS candidates across cycles (minimal cols only)
    oos_selected_rows = []  # ML-selected across cycles
    cycle_summaries = []


    for _, cy in cycles_df.iterrows():
        c = int(cy["cycle"])
        is_from, is_to = cy["IS_from"], cy["IS_to"]
        oos_from, oos_to = cy["OoS_from"], cy["OoS_to"]

        # -------------------------
        # IS: filter by close_dt (known outcomes)
        # -------------------------
        df_is = df[
            (df["close_date"] >= is_from.date()) &
            (df["close_date"] <= is_to.date())
        ].copy()


        # -------------------------
        # OoS candidates: filter by open_dt (decisions)
        # -------------------------
        df_oos = df[
            (df["open_date"] >= oos_from.date()) &
            (df["open_date"] <= oos_to.date())
        ].copy()


        if df_is.empty or df_oos.empty:
            if params.verbose_cycles:
                print(f"\nCycle {c}: IS[{is_from.date()}→{is_to.date()}] OoS[{oos_from.date()}→{oos_to.date()}]  "
                      f"IS_rows={len(df_is)} OoS_rows={len(df_oos)}  -> SKIP (empty slice)")
            cycle_summaries.append(
                dict(cycle=c, IS_rows=len(df_is), OoS_rows=len(df_oos), selected_rows=0, pnlR_sum=0.0, winrate=np.nan)
            )
            continue
        
        # Baseline accumulator: store minimal OoS candidate rows (no features)
        # Baseline accumulator: store OoS candidate rows needed for metrics
        oos_all_rows.append(
            df_oos[["strategy_uid", "open_dt", "close_dt", PNL_COL, PNLR_COL, PREMIUM_COL]].copy()
        )


        
        # Train
        X_is = df_is[FEATURE_COLS]
        y_is = df_is[TARGET_COL].astype(int)

        model = LGBMClassifier(**params.lgbm_params)
        model.fit(X_is, y_is)

        # Predict probability on OoS candidates
        X_oos = df_oos[FEATURE_COLS]
        proba = model.predict_proba(X_oos)[:, 1]
        df_oos["p_pred"] = proba

        # Selection: top K per open_date

        # Rank STRATEGIES (not trades) per day:
        # score per (open_date, strategy_uid) = max predicted probability among that strategy's trades that day
        strat_scores = (
            df_oos.groupby(["open_date", "strategy_uid"], as_index=False)["p_pred"]
                 .max()
                 .rename(columns={"p_pred": "strategy_score"})
        )

        # Pick top-K strategies per day (K is "top strategies per day")
        top_strats = (
            strat_scores.sort_values(["open_date", "strategy_score"], ascending=[True, False])
                        .groupby("open_date", as_index=False)
                        .head(params.top_k_per_day)
        )

        # Keep ALL trades for the selected strategies on that day
        selected = df_oos.merge(
            top_strats[["open_date", "strategy_uid"]],
            on=["open_date", "strategy_uid"],
            how="inner",
        ).copy()


        # Diagnostic snapshot match
        # print(
        #     f"CHECK {oos_from.date()} → {oos_to.date()} | "
        #     f"OoS_rows={len(df_oos)} | "
        #     f"Snapshot_match="
        #     f"{len(df[(df['open_date']>=oos_from.date()) & (df['open_date']<=oos_to.date())])}"
        # )

        # Entry-window metrics (quality)
        pnlR_sum = float(selected[PNLR_COL].sum())
        pnlR_mean = float(selected[PNLR_COL].mean()) if len(selected) else 0.0
        winrate = float((selected[PNLR_COL] > 0).mean()) if len(selected) else np.nan

        # Append to global selected set (dedupe by strategy_uid + open_dt)
        oos_selected_rows.append(selected)

        if params.verbose_cycles:
            print(
                f"\nCycle {c}: "
                f"IS_close[{is_from.date()}→{is_to.date()}] rows={len(df_is)}  "
                f"OoS_open[{oos_from.date()}→{oos_to.date()}] rows={len(df_oos)}  "
                f"Selected={len(selected)}  pnl_R(sum)={pnlR_sum:.4f}  winrate={winrate:.2%}"
            )

        cycle_summaries.append(
            dict(
                cycle=c,
                IS_from=str(is_from.date()),
                IS_to=str(is_to.date()),
                OoS_from=str(oos_from.date()),
                OoS_to=str(oos_to.date()),
                IS_rows=int(len(df_is)),
                OoS_rows=int(len(df_oos)),
                selected_rows=int(len(selected)),
                pnlR_sum=pnlR_sum,
                pnlR_mean=pnlR_mean,
                winrate=winrate,
            )
        )

    # -------------------------
    # Stitch BASELINE (all OoS candidates across cycles)
    # -------------------------
    if oos_all_rows:
        base_all = pd.concat(oos_all_rows, ignore_index=True)
        # Do NOT deduplicate: multi-entry strategies are normal
        base_all = base_all.sort_values(["close_dt", "open_dt", "strategy_uid"]).reset_index(drop=True)
    else:
        base_all = pd.DataFrame()
        
    print("DEBUG baseline: stitched rows =", len(base_all))
    print("DEBUG baseline: open_dt min/max =", base_all["open_dt"].min(), base_all["open_dt"].max())

    # -------------------------
    # Stitch SELECTED (ML)
    # -------------------------
    if oos_selected_rows:
        sel_all = pd.concat(oos_selected_rows, ignore_index=True)
        # Do NOT deduplicate: multi-entry strategies are normal
        sel_all = sel_all.sort_values(["close_dt", "open_dt", "strategy_uid"]).reset_index(drop=True)
    else:
        sel_all = pd.DataFrame()

    # Realized series (by close_dt) for both
    base_daily, base_monthly = _build_realized(base_all, pnl_col=PNL_COL, pnlr_col=PNLR_COL)
    sel_daily, sel_monthly = _build_realized(sel_all, pnl_col=PNL_COL, pnlr_col=PNLR_COL)

    # Metrics
    baseline_metrics = _compute_metrics(base_all, initial_equity=float(params.initial_equity))
    ml_metrics = _compute_metrics(sel_all, initial_equity=float(params.initial_equity))

    # Keep the old prints but make them comparative
    print("\nFINAL (stitched by close_dt):")
    print(f" BASELINE trades total: {baseline_metrics['trades']}")
    print(f" BASELINE total pnl_R:  {baseline_metrics['total_pnlR']:.4f}")
    print(f" BASELINE max DD $:     {baseline_metrics['max_dd_$']:.2f}   max DD %: {baseline_metrics['max_dd_%']:.2%}")

    print(f" ML trades total:       {ml_metrics['trades']}")
    print(f" ML total pnl_R:        {ml_metrics['total_pnlR']:.4f}")
    print(f" ML max DD $:           {ml_metrics['max_dd_$']:.2f}   max DD %: {ml_metrics['max_dd_%']:.2%}")


    return {
        "params": params,
        "cycles": cycles_df,
        "cycle_summaries": pd.DataFrame(cycle_summaries),

        "baseline_trades": base_all,
        "selected_trades": sel_all,

        "baseline_realized_daily": base_daily,
        "baseline_realized_monthly": base_monthly,
        "ml_realized_daily": sel_daily,
        "ml_realized_monthly": sel_monthly,

        "baseline_metrics": baseline_metrics,
        "ml_metrics": ml_metrics,
    }



# -------------------------
# Spyder harness
# -------------------------
if __name__ == "__main__":
    # Point this to one of your built panel CSVs from ml/datasets/
    DATASET = r"C:\Users\mauro\MAURO\Spyder\Portfolio26\ml\datasets\panel__SNAP_20251218_123158__N11.csv"

    p = RunParams(
        dataset_csv_path=DATASET,
        start_date=None,
        end_date=None,
        is_months=2,
        oos_months=1,
        anchored_type="U",
        step_months=1,
        top_k_per_day=3,
        verbose_cycles=True,
    )

    out = run_fwa_single(p)
    print("\nCycle summary head:")
    print(out["cycle_summaries"].head(10).to_string(index=False))
