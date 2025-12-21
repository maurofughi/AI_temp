# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 12:15:55 2025

@author: mauro


Portfolio26 - Phase 3 ML
ml2.py

Weekly CPO (approx) Walk-Forward Analysis (FWA) using LightGBM classifier.

Key differences vs ml1.py:
- IS window length remains in MONTHS, but cadence is WEEKLY (end-of-week Friday).
- OoS window is in WEEKS (e.g., 1 week ahead).
- Step is in WEEKS (must be >= OoS weeks to avoid overlap double-counting).
- Selection is Top-K STRATEGIES per DAY (not trades).
- Builds a synthetic OoS prediction panel by re-applying estimated market features
  + strategy typical features (from IS) across OoS trading days.

Leakage rules preserved:
- IS training slice filtered by close_dt (information availability)
- OoS candidates filtered by open_dt (decision time)
- Equity stitched by close_dt (real DD/path)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier
except Exception as e:
    LGBMClassifier = None
    _LGBM_IMPORT_ERROR = e
    
# ---------------------------------------------------------------------
# Global RNG seed for feature randomization (price, VIX, gap, etc.)
# ---------------------------------------------------------------------
FEATURE_RANDOM_SEED = 354  # change this if you want different random paths


# -------------------------
# Parameters
# -------------------------
@dataclass
class RunParamsWeekly:
    dataset_csv_path: str

    # optional overall bounds (date-only strings "YYYY-MM-DD")
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # IS window length (months) - unchanged conceptually
    is_months: int = 2

    # OoS cadence/length (weeks)
    oos_weeks: int = 1
    step_weeks: int = 1

    anchored_type: str = "U"   # "U" (Unanchored) or "A" (Anchored)

    # selection
    top_k_per_day: int = 3

    # model params
    lgbm_params: Dict[str, Any] = None

    verbose_cycles: bool = True
    initial_equity: float = 100000.0
    
    # diagnostics
    debug_cycle_to_print: Optional[int] = 10   # e.g., 10 to print cycle 10
    debug_max_rows: int = 40                     # cap printed rows per table



# Keep consistent with ml1.py columns
FEATURE_COLS = [
    "dow",
    "open_minute",
    "opening_price",
    "premium",
    "margin_req",
    "opening_vix",
    "gap",
]
TARGET_COL = "label"
PNL_COL = "pnl"
PNLR_COL = "pnl_R"
PREMIUM_COL = "premium"


# -------------------------
# Utils: date parsing
# -------------------------
def _to_ts_date(d: str) -> pd.Timestamp:
    # normalize to date-only midnight
    return pd.Timestamp(d).normalize()


def _date_floor_from_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _make_week_table(trading_dates: pd.Series) -> pd.DataFrame:
    """
    trading_dates: Series[Timestamp normalized] unique trading days.
    Returns weeks with week_id (period W-FRI) and actual week_start/week_end (trading days).
    """
    d = pd.Series(pd.to_datetime(trading_dates, errors="coerce")).dropna().dt.normalize()
    if d.empty:
        return pd.DataFrame(columns=["week_id", "week_start", "week_end"])

    tmp = pd.DataFrame({"d": d.unique()})
    tmp = tmp.sort_values("d").reset_index(drop=True)
    tmp["week_id"] = tmp["d"].dt.to_period("W-FRI").astype(str)

    weeks = (
        tmp.groupby("week_id", as_index=False)
           .agg(week_start=("d", "min"), week_end=("d", "max"))
           .sort_values("week_start")
           .reset_index(drop=True)
    )
    return weeks


def _compute_cycles_weekly(
    params: RunParamsWeekly,
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    weeks_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build weekly cycles:

    - OoS is defined by week blocks (W-FRI groups) on OPEN dates.
    - IS_end = day before OoS_start.
    - IS_start:
        Anchored: fixed at overall start
        Unanchored: IS_start = IS_end - is_months + 1 day (month-offset)
    """
    if params.step_weeks < params.oos_weeks:
        raise ValueError("Invalid parameters: step_weeks must be >= oos_weeks to avoid overlapping OoS windows.")

    # Overall bounds
    start = _to_ts_date(params.start_date) if params.start_date else data_start.normalize()
    end = _to_ts_date(params.end_date) if params.end_date else data_end.normalize()

    # clamp to available data
    start = max(start, data_start.normalize())
    end = min(end, data_end.normalize())

    if weeks_df.empty:
        return pd.DataFrame(columns=["cycle", "IS_from", "IS_to", "OoS_from", "OoS_to"])

    # Find the first OoS week such that IS_end has at least is_months history from start
    earliest_is_end = (start + pd.DateOffset(months=params.is_months)) - pd.Timedelta(days=1)

    # OoS_start is week_start; IS_end is OoS_start-1
    # Need IS_end >= earliest_is_end  => OoS_start >= earliest_is_end + 1
    first_oos_start_min = earliest_is_end + pd.Timedelta(days=1)

    # choose first week whose week_start >= that threshold
    candidate_idx = weeks_df.index[weeks_df["week_start"] >= first_oos_start_min]
    if len(candidate_idx) == 0:
        return pd.DataFrame(columns=["cycle", "IS_from", "IS_to", "OoS_from", "OoS_to"])

    i0 = int(candidate_idx[0])

    cycles = []
    anchored_is_from = start

    i = i0
    safety = 0
    while i < len(weeks_df):
        oos_slice = weeks_df.iloc[i:i + params.oos_weeks]
        if oos_slice.empty:
            break

        oos_from = pd.Timestamp(oos_slice["week_start"].min()).normalize()
        oos_to = pd.Timestamp(oos_slice["week_end"].max()).normalize()

        if oos_from > end:
            break
        if oos_to > end:
            oos_to = end

        is_to = (oos_from - pd.Timedelta(days=1)).normalize()

        if params.anchored_type.upper() == "A":
            is_from = anchored_is_from
        else:
            is_from = (is_to - pd.DateOffset(months=params.is_months) + pd.Timedelta(days=1)).normalize()
            if is_from < start:
                is_from = start

        cycles.append(
            {
                "cycle": len(cycles) + 1,
                "IS_from": is_from,
                "IS_to": is_to,
                "OoS_from": oos_from,
                "OoS_to": oos_to,
            }
        )

        # advance
        i += params.step_weeks
        safety += 1
        if safety > 2000:
            raise RuntimeError("Too many cycles generated; check weekly parameters.")

    return pd.DataFrame(cycles)


# -------------------------
# Metrics (copied from ml1.py, corrected PCR usage)
# -------------------------
def _annualized_sharpe(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    if daily_returns is None or len(daily_returns) < 2:
        return np.nan
    mu = float(daily_returns.mean())
    sd = float(daily_returns.std(ddof=1))
    if sd == 0:
        return np.nan
    return float((mu / sd) * np.sqrt(periods_per_year))


def _build_realized(trades: pd.DataFrame, pnl_col: str, pnlr_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def _pcr_from_pnl_and_premium(pnl: pd.Series, premium: pd.Series) -> float:
    if pnl is None or pnl.empty or premium is None or premium.empty:
        return np.nan
    total_pnl = float(pnl.sum())
    total_abs_prem = float(premium.abs().sum())
    if total_abs_prem == 0.0:
        return np.nan
    return total_pnl / total_abs_prem  # fraction


def _compute_metrics(trades: pd.DataFrame, initial_equity: float) -> Dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "trades": 0,
            "total_pnl_$": 0.0,
            "total_pnlR": 0.0,
            "return_%": 0.0,
            "PCR": np.nan,
            "max_dd_$": 0.0,
            "max_dd_%": 0.0,
            "sharpe_daily": np.nan,
            "win_month_%": np.nan,
            "avg_month_pnl_$": np.nan,
            "median_month_pnl_$": np.nan,
            "best_month_pnl_$": np.nan,
            "worst_month_pnl_$": np.nan,
        }

    daily, monthly = _build_realized(trades, pnl_col=PNL_COL, pnlr_col=PNLR_COL)

    eq = float(initial_equity) + daily[PNL_COL].cumsum().to_numpy()
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([float(initial_equity)])
    dd = eq - peak

    max_dd_dollar = float(dd.min()) if len(dd) else 0.0
    max_dd_pct = float((dd / peak).min()) if len(dd) else 0.0  # negative fraction

    daily_ret = daily[PNL_COL] / float(initial_equity)
    sharpe = _annualized_sharpe(daily_ret)

    if monthly.empty:
        win_month_pct = np.nan
        avg_month = np.nan
        med_month = np.nan
        best_month = np.nan
        worst_month = np.nan
    else:
        m = monthly[PNL_COL]
        win_month_pct = float((m > 0).mean())
        avg_month = float(m.mean())
        med_month = float(m.median())
        best_month = float(m.max())
        worst_month = float(m.min())

    total_pnl = float(trades[PNL_COL].sum())
    total_pnlR = float(trades[PNLR_COL].sum())
    ret_pct = float(total_pnl / float(initial_equity))

    return {
        "trades": int(len(trades)),
        "total_pnl": total_pnl,
        "total_pnlR": total_pnlR,
        "return_pct": ret_pct,
        "pcr": _pcr_from_pnl_and_premium(trades[PNL_COL], trades[PREMIUM_COL]),
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
# Weekly CPO prediction panel builder
# -------------------------
def _estimate_market_features_from_is(
    df_is: pd.DataFrame,
    is_end: pd.Timestamp,
    atr_lookback_days: int = 10,
) -> Dict[str, float]:
    """
    Estimate market-level features for the *next* week from the IS slice.

    Rules (as agreed):

    - opening_price:
        • Use the last `atr_lookback_days` trading days in IS (by open_date).
        • Take ALL opening_price values over that window.
        • mean  = average of those prices
        • std   = std of those prices  (proxy for ATR in price terms).

    - opening_vix:
        • Take the last calendar month inside IS (from first-of-month up to `is_end`).
        • Compute mean VIX over that window.
        • Set sigma as:
              mean >= 25        →  8% of mean
              17 <= mean < 25   →  6% of mean
              0 < mean < 17     →  9% of mean

    - gap:
        • Use *all* IS rows to compute mean and std of gap.
        • Later we will draw from N(mean, std) per OoS day.

    Returns:
        {
            "price_mean", "price_std",
            "vix_mean", "vix_std",
            "gap_mean", "gap_std",
        }
    """
    if df_is.empty:
        return {
            "price_mean": np.nan,
            "price_std": 0.0,
            "vix_mean": np.nan,
            "vix_std": 0.0,
            "gap_mean": 0.0,
            "gap_std": 0.0,
        }

    tmp = df_is.copy()
    tmp["open_date"] = tmp["open_dt"].dt.normalize()

    # --- opening_price: use last N trading days in IS
    dates_all = (
        tmp["open_date"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if dates_all:
        if len(dates_all) > atr_lookback_days:
            last_dates = dates_all[-atr_lookback_days:]
        else:
            last_dates = dates_all

        sl_price = tmp[tmp["open_date"].isin(last_dates)]
        price_series = sl_price["opening_price"].dropna() if "opening_price" in sl_price.columns else pd.Series([], dtype=float)
        if not price_series.empty:
            price_mean = float(price_series.mean())
            price_std = float(price_series.std(ddof=0))
        else:
            price_mean = np.nan
            price_std = 0.0
    else:
        price_mean = np.nan
        price_std = 0.0

    # --- opening_vix: last calendar month within IS
    is_end_date = pd.to_datetime(is_end).normalize()
    try:
        last_month_start = is_end_date.replace(day=1)
    except Exception:
        last_month_start = is_end_date

    mask_vix = (tmp["open_date"] >= last_month_start) & (tmp["open_date"] <= is_end_date)
    vix_series = tmp.loc[mask_vix, "opening_vix"].dropna() if "opening_vix" in tmp.columns else pd.Series([], dtype=float)

    if vix_series.empty and "opening_vix" in tmp.columns:
        # fallback: all IS VIX values
        vix_series = tmp["opening_vix"].dropna()

    if not vix_series.empty:
        vix_mean = float(vix_series.mean())
        if vix_mean >= 25.0:
            sigma_factor = 0.08
        elif vix_mean >= 17.0:
            sigma_factor = 0.06
        elif vix_mean > 0.0:
            sigma_factor = 0.09
        else:
            sigma_factor = 0.0
        vix_std = float(vix_mean * sigma_factor)
    else:
        vix_mean = np.nan
        vix_std = 0.0

    # --- gap: full IS distribution
    if "gap" in tmp.columns:
        gap_series = tmp["gap"].dropna()
    else:
        gap_series = pd.Series([], dtype=float)

    if not gap_series.empty:
        gap_mean = float(gap_series.mean())
        gap_std = float(gap_series.std(ddof=0))
    else:
        gap_mean = 0.0
        gap_std = 0.0

    return {
        "price_mean": price_mean,
        "price_std": price_std,
        "vix_mean": vix_mean,
        "vix_std": vix_std,
        "gap_mean": gap_mean,
        "gap_std": gap_std,
    }




def _strategy_typicals_from_is(df_is: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy-specific typical features taken from IS.

    Rules:
    - open_minute: FIRST entry time in IS (min over open_minute).
    - premium, margin_req: median over IS (per strategy).
    """
    g = df_is.groupby("strategy_uid", as_index=False).agg(
        open_minute=("open_minute", "min"),
        premium=("premium", "median"),
        margin_req=("margin_req", "median"),
    )
    return g



def _build_oos_prediction_panel(
    df_all: pd.DataFrame,
    df_is: pd.DataFrame,
    oos_days: List[pd.Timestamp],
    strategy_uids: List[str],
    is_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Create synthetic OoS prediction rows for (day, strategy) using weekly CPO logic.

    Features:
      - dow: from OoS day
      - open_minute, premium, margin_req: per-strategy typics from IS
      - opening_price: random draw from N(price_mean, price_std) where
            price_mean/std are computed from last N trading days in IS
      - opening_vix: random draw from N(vix_mean, vix_std) where
            vix_mean = mean VIX over the last calendar month in IS
            vix_std  = mean * {8%, 6%, 9%} depending on the band
      - gap: random draw from N(gap_mean, gap_std) computed over full IS
    """
    # Market-level estimates from IS
    market = _estimate_market_features_from_is(df_is, is_end=is_end)

    typ = _strategy_typicals_from_is(df_is)
    # Make sure all strategies in this OoS week appear (even if some had no IS rows)
    typ = typ.set_index("strategy_uid").reindex(strategy_uids).reset_index()

    rows = []
    # Keep dates ordered for determinism
    for d in sorted(oos_days):
        d_norm = d.normalize()
        dow = int(d.dayofweek)

        # --- opening_price: daily draw
        price_mean = market.get("price_mean", np.nan)
        price_std = market.get("price_std", 0.0)
        if pd.notna(price_mean) and price_std > 0:
            day_price = float(np.random.normal(price_mean, price_std))
        else:
            day_price = float(price_mean) if pd.notna(price_mean) else np.nan

        # --- opening_vix: daily draw
        vix_mean = market.get("vix_mean", np.nan)
        vix_std = market.get("vix_std", 0.0)
        if pd.notna(vix_mean) and vix_std > 0:
            day_vix = float(np.random.normal(vix_mean, vix_std))
        else:
            day_vix = float(vix_mean) if pd.notna(vix_mean) else np.nan

        # --- gap: daily draw from IS distribution
        gap_mean = market.get("gap_mean", 0.0)
        gap_std = market.get("gap_std", 0.0)
        if gap_std > 0:
            day_gap = float(np.random.normal(gap_mean, gap_std))
        else:
            day_gap = float(gap_mean)

        for _, r in typ.iterrows():
            rows.append(
                {
                    "open_date": d_norm,
                    "strategy_uid": r["strategy_uid"],
                    "dow": dow,
                    "open_minute": r["open_minute"],
                    "premium": r["premium"],
                    "margin_req": r["margin_req"],
                    "opening_price": day_price,
                    "opening_vix": day_vix,
                    "gap": day_gap,
                }
            )

    panel = pd.DataFrame(rows)

    # Ensure all feature cols exist in the right shape
    for c in FEATURE_COLS:
        if c not in panel.columns:
            panel[c] = np.nan

    return panel



#---------------DIAGNOSTIC PRINTOUTS CYCLES HELPERS
def _print_df(title: str, df: pd.DataFrame, max_rows: int = 40) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    if df is None:
        print("<None>")
        return
    if df.empty:
        print("<EMPTY>")
        return

    # Avoid pandas truncation surprises in console
    with pd.option_context(
        "display.max_rows", max_rows,
        "display.max_columns", 200,
        "display.width", 200,
        "display.max_colwidth", 80,
    ):
        print(df.head(max_rows).to_string(index=False))
    if len(df) > max_rows:
        print(f"... ({len(df) - max_rows} more rows not shown)")


# -------------------------
# Core weekly FWA run
# -------------------------
def run_fwa_weekly(params: RunParamsWeekly) -> Dict[str, Any]:
    if LGBMClassifier is None:
        raise ImportError(f"LightGBM is not available: {_LGBM_IMPORT_ERROR}")

    if params.lgbm_params is None:
        params.lgbm_params = dict(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        
    np.random.seed(FEATURE_RANDOM_SEED)
    
    df = pd.read_csv(params.dataset_csv_path)
    df["open_dt"] = pd.to_datetime(df["open_dt"], errors="coerce")
    df["close_dt"] = pd.to_datetime(df["close_dt"], errors="coerce")

    needed = ["open_dt", "close_dt", "strategy_uid"] + FEATURE_COLS + [TARGET_COL, PNL_COL, PNLR_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # enforce structural fields only (no feature dropping)
    df = df.dropna(subset=["open_dt", "close_dt", TARGET_COL, PNLR_COL, PNL_COL]).copy()
    df = df.sort_values("open_dt").reset_index(drop=True)

    data_start = df["open_dt"].min().normalize()
    data_end = df["open_dt"].max().normalize()

    # Build trading week table from open dates (decision calendar)
    open_dates = df["open_dt"].dt.normalize().dropna().drop_duplicates().sort_values()
    weeks_df = _make_week_table(open_dates)

    cycles_df = _compute_cycles_weekly(params, data_start, data_end, weeks_df)

    print("\nCYCLES (weekly cadence; W-FRI blocks on OPEN dates):")
    if cycles_df.empty:
        print("No cycles could be created with the current parameters/date range.")
    else:
        print(cycles_df.to_string(index=False))

    print("\nRUN PARAMS (WEEKLY CPO):")
    print(f" dataset: {params.dataset_csv_path}")
    print(f" anchored_type: {params.anchored_type}  IS={params.is_months} months  OoS={params.oos_weeks} weeks  step={params.step_weeks} weeks")
    print(f" top_k_per_day: {params.top_k_per_day}")
    print(f" features: {FEATURE_COLS}")
    print(f" target: {TARGET_COL} (label = pnl_R > 0)")

    oos_all_rows = []
    oos_selected_rows = []
    cycle_summaries = []

    if cycles_df.empty:
        base_all = pd.DataFrame()
        sel_all = pd.DataFrame()
        return {
            "params": params,
            "cycles": cycles_df,
            "cycle_summaries": pd.DataFrame(),
            "baseline_trades": base_all,
            "selected_trades": sel_all,
            "baseline_metrics": _compute_metrics(base_all, params.initial_equity),
            "ml_metrics": _compute_metrics(sel_all, params.initial_equity),
        }

    for _, cy in cycles_df.iterrows():
        c = int(cy["cycle"])
        is_from, is_to = cy["IS_from"], cy["IS_to"]
        oos_from, oos_to = cy["OoS_from"], cy["OoS_to"]

        # IS: outcomes known by close_dt
        df_is = df[(df["close_dt"].dt.normalize() >= is_from) & (df["close_dt"].dt.normalize() <= is_to)].copy()

        # OoS actual candidates: decisions by open_dt
        df_oos_actual = df[(df["open_dt"].dt.normalize() >= oos_from) & (df["open_dt"].dt.normalize() <= oos_to)].copy()
        
        # -------------------------
        # DEBUG: print IS slice and OoS actual candidates (for ONE selected cycle)
        # -------------------------
        if params.debug_cycle_to_print is not None and c == int(params.debug_cycle_to_print):
            cols_is = ["strategy_uid", "open_dt", "close_dt"] + FEATURE_COLS + [TARGET_COL, PNL_COL, PNLR_COL]
            cols_is = [x for x in cols_is if x in df_is.columns]
        
            cols_oos = ["strategy_uid", "open_dt", "close_dt", PNL_COL, PNLR_COL, PREMIUM_COL]
            cols_oos = [x for x in cols_oos if x in df_oos_actual.columns]
        
            _print_df(
                f"DEBUG Cycle {c} | IS TRAIN SLICE (filtered by close_dt) | "
                f"{is_from.date()} → {is_to.date()} | rows={len(df_is)} | cols={cols_is}",
                df_is[cols_is].sort_values(["close_dt", "open_dt", "strategy_uid"]).tail(params.debug_max_rows),
                max_rows=int(params.debug_max_rows),
            )
        
            _print_df(
                f"DEBUG Cycle {c} | OoS ACTUAL CANDIDATES (filtered by open_dt) | "
                f"{oos_from.date()} → {oos_to.date()} | rows={len(df_oos_actual)}",
                df_oos_actual[cols_oos].sort_values(["open_dt", "strategy_uid"]),
                max_rows=int(params.debug_max_rows),
            )
        #-----------------------------------------------------------------------
        
        
        if df_is.empty or df_oos_actual.empty:
            if params.verbose_cycles:
                print(
                    f"\nCycle {c}: IS_close[{is_from.date()}→{is_to.date()}] rows={len(df_is)}  "
                    f"OoS_open[{oos_from.date()}→{oos_to.date()}] rows={len(df_oos_actual)} -> SKIP"
                )
            cycle_summaries.append(
                dict(cycle=c, IS_rows=len(df_is), OoS_rows=len(df_oos_actual), selected_rows=0, pnlR_sum=0.0, winrate=np.nan)
            )
            continue

        # Baseline: store ALL OoS actual trades (minimal cols) for this cycle
        oos_all_rows.append(
            df_oos_actual[["strategy_uid", "open_dt", "close_dt", PNL_COL, PNLR_COL, PREMIUM_COL]].copy()
        )

        # Train model on IS
        X_is = df_is[FEATURE_COLS]
        y_is = df_is[TARGET_COL].astype(int)

        model = LGBMClassifier(**params.lgbm_params)
        model.fit(X_is, y_is)

        # Build synthetic OoS prediction panel (daily x strategy)
        oos_days = (
            df_oos_actual["open_dt"].dt.normalize()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        strategy_uids = sorted(df_oos_actual["strategy_uid"].dropna().unique().tolist())

        pred_panel = _build_oos_prediction_panel(
            df_all=df,
            df_is=df_is,
            oos_days=oos_days,
            strategy_uids=strategy_uids,
            is_end=is_to,
        )

        # Predict per (day, strategy)
        proba = model.predict_proba(pred_panel[FEATURE_COLS])[:, 1]
        pred_panel["p_pred"] = proba
        
        # -------------------------
        # DEBUG: print the synthetic OoS prediction panel + per-day Top-K strategies
        # -------------------------
        if params.debug_cycle_to_print is not None and c == int(params.debug_cycle_to_print):
            cols_pred = ["open_date", "strategy_uid"] + FEATURE_COLS + ["p_pred"]
            cols_pred = [x for x in cols_pred if x in pred_panel.columns]
        
            _print_df(
                f"DEBUG Cycle {c} | OoS PREDICTION PANEL fed to predict_proba() | "
                f"days={len(oos_days)} | strategies={len(strategy_uids)} | rows={len(pred_panel)}",
                pred_panel[cols_pred].sort_values(["open_date", "p_pred"], ascending=[True, False]),
                max_rows=int(params.debug_max_rows),
            )
        # -------------------------
        
        
        # Top-K strategies PER DAY (not trades)
        pred_panel["open_date"] = pred_panel["open_date"].dt.normalize()
        top_strats = (
            pred_panel.sort_values(["open_date", "p_pred"], ascending=[True, False])
                      .groupby("open_date", as_index=False)
                      .head(int(params.top_k_per_day))
        )
        
        #---------DIAGNOSTIC------------
        if params.debug_cycle_to_print is not None and c == int(params.debug_cycle_to_print):
            _print_df(
                f"DEBUG Cycle {c} | Top-{params.top_k_per_day} STRATEGIES PER DAY (from synthetic panel)",
                top_strats[["open_date", "strategy_uid", "p_pred"]].sort_values(["open_date", "p_pred"], ascending=[True, False]),
                max_rows=int(params.debug_max_rows),
            )
        #----------------------

        # Select ALL actual trades for those strategies on those days
        df_oos_actual["open_date"] = df_oos_actual["open_dt"].dt.normalize()
        key = top_strats[["open_date", "strategy_uid"]].copy()
        key["sel_flag"] = 1

        selected = df_oos_actual.merge(key, on=["open_date", "strategy_uid"], how="inner")
        # keep only the core cols for global stitching
        selected_core = selected[["strategy_uid", "open_dt", "close_dt", PNL_COL, PNLR_COL, PREMIUM_COL]].copy()

        oos_selected_rows.append(selected_core)

        pnlR_sum = float(selected_core[PNLR_COL].sum()) if len(selected_core) else 0.0
        pnlR_mean = float(selected_core[PNLR_COL].mean()) if len(selected_core) else 0.0
        winrate = float((selected_core[PNLR_COL] > 0).mean()) if len(selected_core) else np.nan

        if params.verbose_cycles:
            print(
                f"\nCycle {c}: "
                f"IS_close[{is_from.date()}→{is_to.date()}] rows={len(df_is)}  "
                f"OoS_open[{oos_from.date()}→{oos_to.date()}] rows={len(df_oos_actual)}  "
                f"TopK_strats/day={params.top_k_per_day}  "
                f"Selected_trades={len(selected_core)}  pnl_R(sum)={pnlR_sum:.4f}  winrate={winrate:.2%}"
            )

        cycle_summaries.append(
            dict(
                cycle=c,
                IS_from=str(is_from.date()),
                IS_to=str(is_to.date()),
                OoS_from=str(oos_from.date()),
                OoS_to=str(oos_to.date()),
                IS_rows=int(len(df_is)),
                OoS_rows=int(len(df_oos_actual)),
                selected_rows=int(len(selected_core)),
                pnlR_sum=pnlR_sum,
                pnlR_mean=pnlR_mean,
                winrate=winrate,
            )
        )

    # Stitch baseline + selected (no dedupe; multi-entry is normal)
    base_all = pd.concat(oos_all_rows, ignore_index=True) if oos_all_rows else pd.DataFrame()
    sel_all = pd.concat(oos_selected_rows, ignore_index=True) if oos_selected_rows else pd.DataFrame()

    if not base_all.empty:
        base_all = base_all.sort_values(["close_dt", "open_dt", "strategy_uid"]).reset_index(drop=True)
    if not sel_all.empty:
        sel_all = sel_all.sort_values(["close_dt", "open_dt", "strategy_uid"]).reset_index(drop=True)

    baseline_metrics = _compute_metrics(base_all, initial_equity=float(params.initial_equity))
    ml_metrics = _compute_metrics(sel_all, initial_equity=float(params.initial_equity))

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
        "baseline_metrics": baseline_metrics,
        "ml_metrics": ml_metrics,
    }


# -------------------------
# Spyder harness
# -------------------------
if __name__ == "__main__":
    DATASET = r"C:\Users\mauro\MAURO\Spyder\Portfolio26\ml\datasets\panel__SNAP_YYYYMMDD_HHMMSS__N9.csv"

    p = RunParamsWeekly(
        dataset_csv_path=DATASET,
        start_date=None,
        end_date=None,
        is_months=2,
        oos_weeks=1,
        step_weeks=1,
        anchored_type="U",
        top_k_per_day=3,
        verbose_cycles=True,
    )

    out = run_fwa_weekly(p)
    print("\nCycle summary head:")
    print(out["cycle_summaries"].head(12).to_string(index=False))
