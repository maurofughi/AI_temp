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

from ml.ml_alloc import apply_max_allocation


try:
    from lightgbm import LGBMClassifier
except Exception as e:
    LGBMClassifier = None
    _LGBM_IMPORT_ERROR = e
    
# ---------------------------------------------------------------------
# Global RNG seed for feature randomization (price, VIX, gap, etc.)
# ---------------------------------------------------------------------
RANDOM_SEED = 113723324  # change this if you want different random paths


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
    
    # ------------------ SELECTION MODE (NEW) ------------------
    # "top_k":   select Top K strategies per day (current behaviour)
    # "bottom_k": drop Bottom K strategies per day (avoid worst)
    selection_mode: str = "top_k"
    # ---------------------------------------------------------

    # selection
    top_k_per_day: int = 3

    # model params
    lgbm_params: Dict[str, Any] = None

    verbose_cycles: bool = True
    initial_equity: float = 100000.0
    
    # diagnostics
    debug_cycle_to_print: Optional[int] = 10   # e.g., 10 to print cycle 10
    debug_max_rows: int = 40                     # cap printed rows per table
    
    # allocation / sizing (new)
    # "equal" → current behaviour (1 lot per selected trade)
    # "max_allocation" → call ml_alloc.apply_max_allocation()
    allocation_mode: str = "equal"
    max_allocation: float = 6500.0      # example default; tune as needed
    allocation_tolerance: float = 0.0    # slack on the cap



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


# def _build_equity_dd_series(trades: pd.DataFrame, initial_equity: float) -> pd.DataFrame:
#     """
#     Builds daily equity & drawdown series from realized P&L by close date.
#     Returns a dataframe with:
#       date, pnl, equity, dd, dd_pct
#     """
#     if trades is None or trades.empty:
#         return pd.DataFrame(columns=["date", "pnl", "equity", "dd", "dd_pct"])

#     t = trades.copy()
#     t["close_date"] = t["close_dt"].dt.normalize()

#     daily = (
#         t.groupby("close_date", as_index=False)[PNL_COL]
#          .sum()
#          .rename(columns={"close_date": "date", PNL_COL: "pnl"})
#          .sort_values("date")
#          .reset_index(drop=True)
#     )

#     daily["equity"] = float(initial_equity) + daily["pnl"].cumsum()
#     daily["peak"] = daily["equity"].cummax()
#     daily["dd"] = daily["equity"] - daily["peak"]
#     daily["dd_pct"] = np.where(daily["peak"] != 0, daily["dd"] / daily["peak"], np.nan)

#     return daily[["date", "pnl", "equity", "dd", "dd_pct"]]

def _build_daily_series_with_exposure(trades: pd.DataFrame, initial_equity: float) -> pd.DataFrame:
    """
    Build daily realized P&L plus exposure proxies by close date:
      - pnl_day: sum pnl for trades closing that date
      - uid_day: number of unique strategy_uid among trades closing that date
      - margin_day: sum margin_req among trades closing that date
    Also includes cumulative raw equity and raw dd (optional, but fine to keep).
    """
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["date", "pnl_day", "uid_day", "margin_day"])

    t = trades.copy()
    t["date"] = t["close_dt"].dt.normalize()

    g = t.groupby("date")

    daily = pd.DataFrame({
        "date": g.size().index,
        "pnl_day": g[PNL_COL].sum().values,
        "uid_day": g["strategy_uid"].nunique().values,
        "margin_day": g["margin_req"].sum().values if "margin_req" in t.columns else 0.0,
        "premium_day": g["premium"].sum().values if "premium" in t.columns else 0.0,
    })


    # Safety: avoid zeros
    daily["uid_day"] = daily["uid_day"].replace(0, np.nan)
    daily["margin_day"] = daily["margin_day"].replace(0, np.nan)

    # Keep raw cumulative equity for reference if you want
    daily["equity_raw"] = float(initial_equity) + daily["pnl_day"].cumsum()
    daily["peak_raw"] = daily["equity_raw"].cummax()
    daily["dd_raw"] = daily["equity_raw"] - daily["peak_raw"]
    daily["dd_raw_pct"] = np.where(daily["peak_raw"] != 0, daily["dd_raw"] / daily["peak_raw"], np.nan)
    daily["premium_day"] = daily["premium_day"].replace(0, np.nan)


    return daily


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


def _q(x: pd.Series, q: float) -> float:
    if x is None or x.empty:
        return np.nan
    return float(x.quantile(q))


def _compute_participation_metrics(
    trades: pd.DataFrame,
    initial_equity: float,
    nominal_units: int,
) -> Dict[str, Any]:
    """
    Participation / capacity metrics computed from realized trades.

    We use open_date for participation (what triggers per day).
    We also compute daily sums of margin_req and abs(premium) as a unit-capacity proxy.
    """
    if trades is None or trades.empty:
        return {
            "nominal_units": int(nominal_units),
            "avg_trades_day": np.nan,
            "med_trades_day": np.nan,
            "p95_trades_day": np.nan,
            "max_trades_day": np.nan,
            "avg_unique_uid_day": np.nan,
            "med_unique_uid_day": np.nan,
            "p95_unique_uid_day": np.nan,
            "max_unique_uid_day": np.nan,
            "p95_margin_day": np.nan,
            "max_margin_day": np.nan,
            "p95_abs_premium_day": np.nan,
            "max_abs_premium_day": np.nan,
            "total_pnl_per_nominal_unit": np.nan,
            "total_pnlR_per_nominal_unit": np.nan,
            "total_pnl_per_avg_active_uid": np.nan,
            "total_pnlR_per_avg_active_uid": np.nan,
        }

    t = trades.copy()
    t["open_date"] = t["open_dt"].dt.normalize()

    # trades per day
    trades_day = t.groupby("open_date").size()

    # unique strategy_uids per day
    uid_day = t.groupby("open_date")["strategy_uid"].nunique()

    # daily capacity proxies (require margin_req + premium columns present)
    margin_day = t.groupby("open_date")["margin_req"].sum() if "margin_req" in t.columns else pd.Series(dtype=float)
    abs_prem_day = t.groupby("open_date")["premium"].apply(lambda s: s.abs().sum()) if "premium" in t.columns else pd.Series(dtype=float)

    # totals
    total_pnl = float(t[PNL_COL].sum())
    total_pnlR = float(t[PNLR_COL].sum())

    # normalizations
    nominal_units = max(int(nominal_units), 1)
    avg_active_uid = float(uid_day.mean()) if len(uid_day) else np.nan
    if avg_active_uid and avg_active_uid > 0:
        pnl_per_avg_uid = total_pnl / avg_active_uid
        pnlR_per_avg_uid = total_pnlR / avg_active_uid
    else:
        pnl_per_avg_uid = np.nan
        pnlR_per_avg_uid = np.nan

    return {
        "nominal_units": int(nominal_units),

        "avg_trades_day": float(trades_day.mean()) if len(trades_day) else np.nan,
        "med_trades_day": float(trades_day.median()) if len(trades_day) else np.nan,
        "p95_trades_day": _q(trades_day, 0.95) if len(trades_day) else np.nan,
        "max_trades_day": float(trades_day.max()) if len(trades_day) else np.nan,

        "avg_unique_uid_day": float(uid_day.mean()) if len(uid_day) else np.nan,
        "med_unique_uid_day": float(uid_day.median()) if len(uid_day) else np.nan,
        "p95_unique_uid_day": _q(uid_day, 0.95) if len(uid_day) else np.nan,
        "max_unique_uid_day": float(uid_day.max()) if len(uid_day) else np.nan,

        "p95_margin_day": _q(margin_day, 0.95) if len(margin_day) else np.nan,
        "max_margin_day": float(margin_day.max()) if len(margin_day) else np.nan,

        "p95_abs_premium_day": _q(abs_prem_day, 0.95) if len(abs_prem_day) else np.nan,
        "max_abs_premium_day": float(abs_prem_day.max()) if len(abs_prem_day) else np.nan,

        "total_pnl_per_nominal_unit": total_pnl / nominal_units,
        "total_pnlR_per_nominal_unit": total_pnlR / nominal_units,

        "total_pnl_per_avg_active_uid": pnl_per_avg_uid,
        "total_pnlR_per_avg_active_uid": pnlR_per_avg_uid,
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

    # Global RNG used by feature randomization in the OoS prediction panel
    np.random.seed(int(RANDOM_SEED))

    # Default LGBM parameters if none provided
    if params.lgbm_params is None:
        params.lgbm_params = dict(
            n_estimators=275,
            learning_rate=0.14,
            num_leaves=17,
            min_child_samples=10,
            max_depth=-1,
            subsample=0.12,
            colsample_bytree=0.62,
            reg_alpha=10.0,
            reg_lambda=9.5,
            min_gain_to_split=0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
            # ADD THESE 5 LINES (same as Optuna):
            bagging_seed=RANDOM_SEED,
            feature_fraction_seed=RANDOM_SEED,
            data_random_seed=RANDOM_SEED,
            force_col_wise=True,
            deterministic=True,
        )

    # Load dataset
    df = pd.read_csv(params.dataset_csv_path)
    df["open_dt"] = pd.to_datetime(df["open_dt"], errors="coerce")
    df["close_dt"] = pd.to_datetime(df["close_dt"], errors="coerce")

    needed = ["open_dt", "close_dt", "strategy_uid"] + FEATURE_COLS + [TARGET_COL, PNL_COL, PNLR_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Enforce structural fields only (no feature dropping)
    df = df.dropna(subset=["open_dt", "close_dt", TARGET_COL, PNLR_COL, PNL_COL]).copy()
    df = df.sort_values("open_dt").reset_index(drop=True)

    data_start = df["open_dt"].min().normalize()
    data_end = df["open_dt"].max().normalize()

    # Build trading week table from open dates (decision calendar)
    open_dates = df["open_dt"].dt.normalize().dropna().drop_duplicates().sort_values()
    weeks_df = _make_week_table(open_dates)
    cycles_df = _compute_cycles_weekly(params, data_start, data_end, weeks_df)

    # ------------------------ IS/OoS CYCLES PRINTING --------------------------------------------
    # print("\nCYCLES (weekly cadence; W-FRI blocks on OPEN dates):")
    # if cycles_df.empty:
    #     print("No cycles could be created with the current parameters/date range.")
    # else:
    #     print(cycles_df.to_string(index=False))

    print("\nRUN PARAMS (WEEKLY CPO):")
    print(f" dataset: {params.dataset_csv_path}")
    print(
        f" anchored_type: {params.anchored_type} "
        f"IS={params.is_months} months OoS={params.oos_weeks} weeks step={params.step_weeks} weeks"
    )
    print(f" top_k_per_day: {params.top_k_per_day}")
    print(f" allocation_mode: {params.allocation_mode} "
          f"max_allocation={params.max_allocation} "
          f"tolerance={params.allocation_tolerance}")
    print(f" features: {FEATURE_COLS}")
    print(f" target: {TARGET_COL} (label = pnl_R > 0)")

    oos_all_rows: List[pd.DataFrame] = []
    oos_selected_rows: List[pd.DataFrame] = []
    cycle_summaries: List[Dict[str, Any]] = []
    
    # Quartile diagnostic across cycles (based on ACTUAL P&L and pnl_R)
    quartile_diag_rows = []
    #---


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
        df_is = df[
            (df["close_dt"].dt.normalize() >= is_from)
            & (df["close_dt"].dt.normalize() <= is_to)
        ].copy()

        # OoS actual candidates: decisions by open_dt
        df_oos_actual = df[
            (df["open_dt"].dt.normalize() >= oos_from)
            & (df["open_dt"].dt.normalize() <= oos_to)
        ].copy()

        # -----------------------------------------------------------------------
        # Skip cycle if no IS/OoS rows
        # -----------------------------------------------------------------------
        if df_is.empty or df_oos_actual.empty:
            if params.verbose_cycles:
                print(
                    f"\nCycle {c}: IS_close[{is_from.date()}→{is_to.date()}] rows={len(df_is)} "
                    f"OoS_open[{oos_from.date()}→{oos_to.date()}] rows={len(df_oos_actual)} -> SKIP"
                )
            cycle_summaries.append(
                dict(
                    cycle=c,
                    IS_rows=len(df_is),
                    OoS_rows=len(df_oos_actual),
                    selected_rows=0,
                    pnlR_sum=0.0,
                    winrate=np.nan,
                )
            )
            continue

        
        # ------------------------- QUARTILE DIAGNOSTIC (per OoS week) -------------------------
        # Use ALL OoS candidate trades for this cycle (df_oos_actual),
        # rank by ACTUAL P&L and pnl_R, and compare top vs bottom quartiles.

        diag_df = df_oos_actual.copy()
        n_diag = len(diag_df)

        if n_diag >= 4:
            # --- P&L (dollar) quartiles ---
            q25_pnl = float(diag_df[PNL_COL].quantile(0.25))
            q75_pnl = float(diag_df[PNL_COL].quantile(0.75))

            pnl_bottom = diag_df[diag_df[PNL_COL] <= q25_pnl]
            pnl_top = diag_df[diag_df[PNL_COL] >= q75_pnl]

            pnl_top_mean = float(pnl_top[PNL_COL].mean()) if len(pnl_top) else np.nan
            pnl_bot_mean = float(pnl_bottom[PNL_COL].mean()) if len(pnl_bottom) else np.nan
            pnl_gap = (
                pnl_top_mean - pnl_bot_mean
                if (not np.isnan(pnl_top_mean) and not np.isnan(pnl_bot_mean))
                else np.nan
            )

            # --- pnl_R quartiles ---
            q25_R = float(diag_df[PNLR_COL].quantile(0.25))
            q75_R = float(diag_df[PNLR_COL].quantile(0.75))

            R_bottom = diag_df[diag_df[PNLR_COL] <= q25_R]
            R_top = diag_df[diag_df[PNLR_COL] >= q75_R]

            R_top_mean = float(R_top[PNLR_COL].mean()) if len(R_top) else np.nan
            R_bot_mean = float(R_bottom[PNLR_COL].mean()) if len(R_bottom) else np.nan
            R_gap = (
                R_top_mean - R_bot_mean
                if (not np.isnan(R_top_mean) and not np.isnan(R_bot_mean))
                else np.nan
            )

        else:
            q25_pnl = q75_pnl = np.nan
            pnl_top_mean = pnl_bot_mean = pnl_gap = np.nan
            q25_R = q75_R = np.nan
            R_top_mean = R_bot_mean = R_gap = np.nan

        quartile_diag_rows.append(
            dict(
                cycle=c,
                n_trades=int(n_diag),
                q25_pnl=q25_pnl,
                q75_pnl=q75_pnl,
                q25_R=q25_R,
                q75_R=q75_R,
                pnl_top_mean=pnl_top_mean,
                pnl_bot_mean=pnl_bot_mean,
                pnl_gap=pnl_gap,
                R_top_mean=R_top_mean,
                R_bot_mean=R_bot_mean,
                R_gap=R_gap,
            )
        )

        print(
            f"CYCLE {c} quartiles | n_trades={n_diag} | "
            f"P&L top_mean={pnl_top_mean:.2f} bot_mean={pnl_bot_mean:.2f} gap={pnl_gap:.2f} | "
            f"R top_mean={R_top_mean:.4f} bot_mean={R_bot_mean:.4f} gap={R_gap:.4f}"
        )
        # ----------------------- END QUARTILE DIAGNOSTIC --------------------------------------


        # Baseline: store ALL OoS actual trades (minimal cols) for this cycle
        oos_all_rows.append(
            df_oos_actual[
                ["strategy_uid", "open_dt", "close_dt", PNL_COL, PNLR_COL, PREMIUM_COL, "margin_req"]
            ].copy()
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
        
        # ------------------------- SELECTION MODE ------------------------------
        # Top-K strategies PER DAY (not trades)
        # ------------------------- SELECTION MODE (TOP/BOTTOM K) ------------------------------
        # NOTE:
        #   - Baseline: df_oos_actual (all trades) is unchanged.
        #   - ML: we either KEEP Top K strategies per day ("top_k"),
        #         or DROP Bottom K strategies per day ("bottom_k").

        pred_panel["open_date"] = pred_panel["open_date"].dt.normalize()
        df_oos_actual["open_date"] = df_oos_actual["open_dt"].dt.normalize()

        sel_mode = (params.selection_mode or "top_k").lower()
        k = int(params.top_k_per_day)

        if sel_mode == "bottom_k":
            # -------- BOTTOM-K MODE: REMOVE WORST K STRATEGIES PER DAY --------
            # Sort so lowest p_pred are first within each day.
            bottom_strats = (
                pred_panel.sort_values(
                    ["open_date", "p_pred", "strategy_uid"],
                    ascending=[True, True, True],
                    kind="mergesort",
                )
                .groupby("open_date", as_index=False)
                .head(k)
            )

            # Build mask of (open_date, strategy_uid) to drop
            to_drop = bottom_strats[["open_date", "strategy_uid"]].drop_duplicates()
            to_drop["drop_flag"] = 1

            drop_join = df_oos_actual.merge(
                to_drop, on=["open_date", "strategy_uid"], how="left"
            )

            # Keep all trades EXCEPT those in bottom K per day
            selected = drop_join[drop_join["drop_flag"].isna()].copy()
            selected = selected.drop(columns=["drop_flag"])
        else:
            # -------- TOP-K MODE (DEFAULT): KEEP BEST K STRATEGIES PER DAY --------
            top_strats = (
                pred_panel.sort_values(
                    ["open_date", "p_pred", "strategy_uid"],
                    ascending=[True, False, True],
                    kind="mergesort",
                )
                .groupby("open_date", as_index=False)
                .head(k)
            )

            key = top_strats[["open_date", "strategy_uid", "p_pred"]].copy()
            key["sel_flag"] = 1

            selected = df_oos_actual.merge(
                key, on=["open_date", "strategy_uid"], how="inner"
            )
        # ----------------------- END SELECTION MODE (TOP/BOTTOM K) ----------------------------

        # keep only the core cols for global stitching (plus p_pred if present)
        core_cols = ["strategy_uid", "open_dt", "close_dt", PNL_COL, PNLR_COL, PREMIUM_COL, "margin_req"]
        if "p_pred" in selected.columns:
            core_cols.append("p_pred")
        selected_core = selected[core_cols].copy()

        oos_selected_rows.append(selected_core)

        pnlR_sum = float(selected_core[PNLR_COL].sum()) if len(selected_core) else 0.0
        pnlR_mean = float(selected_core[PNLR_COL].mean()) if len(selected_core) else 0.0
        winrate = (
            float((selected_core[PNLR_COL] > 0).mean())
            if len(selected_core)
            else np.nan
        )

        if params.verbose_cycles:
            print(
                f"\nCycle {c}: "
                f"IS_close[{is_from.date()}→{is_to.date()}] rows={len(df_is)} "
                f"OoS_open[{oos_from.date()}→{oos_to.date()}] rows={len(df_oos_actual)} "
                f"TopK_strats/day={params.top_k_per_day} "
                f"Selected_trades={len(selected_core)} pnl_R(sum)={pnlR_sum:.4f} "
                f"winrate={winrate:.2%}"
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

        # ------------------------- QUARTILE DIAGNOSTIC SUMMARY -------------------------
    quartile_diag = pd.DataFrame(quartile_diag_rows)
    if not quartile_diag.empty:
        print("\nQUARTILE DIAGNOSTIC SUMMARY ACROSS CYCLES (ACTUAL P&L and pnl_R)")
        print(quartile_diag.to_string(index=False))

        n_tot = int(len(quartile_diag))

        # How often top > bottom
        n_pos_pnl = int((quartile_diag["pnl_gap"] > 0).sum())
        n_pos_R = int((quartile_diag["R_gap"] > 0).sum())

        frac_pos_pnl = (n_pos_pnl / n_tot) if n_tot else np.nan
        frac_pos_R = (n_pos_R / n_tot) if n_tot else np.nan

        print(f"\nCycles with P&L top_mean > bottom_mean: {n_pos_pnl}/{n_tot} ({frac_pos_pnl:.2%})")
        print(f"Cycles with R top_mean > bottom_mean:   {n_pos_R}/{n_tot} ({frac_pos_R:.2%})")

        avg_pnl_top = float(quartile_diag["pnl_top_mean"].mean())
        avg_pnl_bot = float(quartile_diag["pnl_bot_mean"].mean())
        avg_pnl_gap = float(quartile_diag["pnl_gap"].mean())

        avg_R_top = float(quartile_diag["R_top_mean"].mean())
        avg_R_bot = float(quartile_diag["R_bot_mean"].mean())
        avg_R_gap = float(quartile_diag["R_gap"].mean())

        print(f"\nAvg P&L top_mean: {avg_pnl_top:.2f}")
        print(f"Avg P&L bot_mean: {avg_pnl_bot:.2f}")
        print(f"Avg P&L gap:      {avg_pnl_gap:.2f}")

        print(f"\nAvg R top_mean:   {avg_R_top:.4f}")
        print(f"Avg R bot_mean:   {avg_R_bot:.4f}")
        print(f"Avg R gap:        {avg_R_gap:.4f}")
    else:
        print("\nQUARTILE DIAGNOSTIC: no usable cycles (too few trades).")
    # -------------------------------------------------------------------------


    # Stitch baseline + selected (no dedupe; multi-entry is normal)
    base_all = pd.concat(oos_all_rows, ignore_index=True) if oos_all_rows else pd.DataFrame()
    sel_all = pd.concat(oos_selected_rows, ignore_index=True) if oos_selected_rows else pd.DataFrame()

    # In max_allocation mode, apply allocation logic to BOTH baseline and ML.
    # - Baseline: allow_extra_lots=False → at most 1 lot per trade under the cap.
    # - ML: allow_extra_lots=True  → 1 lot if possible + extra lots by rank under the cap.
    if params.allocation_mode.lower() == "max_allocation":
        if not base_all.empty:
            base_all = apply_max_allocation(
                trades=base_all,
                max_allocation=float(params.max_allocation),
                margin_tolerance=float(params.allocation_tolerance),
                allow_extra_lots=False,
            )
        if not sel_all.empty:
            sel_all = apply_max_allocation(
                trades=sel_all,
                max_allocation=float(params.max_allocation),
                margin_tolerance=float(params.allocation_tolerance),
                allow_extra_lots=True,
            )

    # Sort after possible allocation
    if not base_all.empty:
        base_all = (
            base_all.sort_values(["close_dt", "open_dt", "strategy_uid"])
            .reset_index(drop=True)
        )

    if not sel_all.empty:
        sel_all = (
            sel_all.sort_values(["close_dt", "open_dt", "strategy_uid"])
            .reset_index(drop=True)
        )


    # Metrics & curves
    baseline_metrics = _compute_metrics(base_all, initial_equity=float(params.initial_equity))
    ml_metrics = _compute_metrics(sel_all, initial_equity=float(params.initial_equity))

    baseline_curve = _build_daily_series_with_exposure(base_all, float(params.initial_equity))
    ml_curve = _build_daily_series_with_exposure(sel_all, float(params.initial_equity))

    # Nominal breadth:
    #   - Baseline nominal = total unique strategies available in the (bounded) dataset used by this run
    #   - ML nominal = Top-K per day (unchanged, BY DESIGN, even if lots>1)
    baseline_nominal = int(df["strategy_uid"].nunique())
    
    if (params.selection_mode or "top_k").lower() == "bottom_k":
        ml_nominal = max(baseline_nominal - int(params.top_k_per_day), 1)
    else:
        ml_nominal = int(params.top_k_per_day)

    baseline_extra = _compute_participation_metrics(
        trades=base_all,
        initial_equity=float(params.initial_equity),
        nominal_units=baseline_nominal,
    )
    ml_extra = _compute_participation_metrics(
        trades=sel_all,
        initial_equity=float(params.initial_equity),
        nominal_units=ml_nominal,
    )

    print("\nFINAL (stitched by close_dt):")
    print(f" BASELINE trades total: {baseline_metrics['trades']}")
    print(f" BASELINE total pnl_R: {baseline_metrics['total_pnlR']:.4f}")
    print(
        f" BASELINE max DD $: {baseline_metrics['max_dd_$']:.2f} "
        f"max DD %: {baseline_metrics['max_dd_%']:.2%}"
    )
    print(f" ML trades total: {ml_metrics['trades']}")
    print(f" ML total pnl_R: {ml_metrics['total_pnlR']:.4f}")
    print(
        f" ML max DD $: {ml_metrics['max_dd_$']:.2f} "
        f"max DD %: {ml_metrics['max_dd_%']:.2%}"
    )

    return {
        "params": params,
        "cycles": cycles_df,
        "cycle_summaries": pd.DataFrame(cycle_summaries),
        "baseline_trades": base_all,
        "selected_trades": sel_all,
        "baseline_metrics": baseline_metrics,
        "ml_metrics": ml_metrics,
        "baseline_extra_metrics": baseline_extra,
        "ml_extra_metrics": ml_extra,
        "baseline_curve": baseline_curve,
        "ml_curve": ml_curve,
    }


# -------------------------
# Spyder harness
# -------------------------
if __name__ == "__main__":
    DATASET = r"C:\Users\mauro\MAURO\Spyder\Portfolio26\ml\datasets\panel__SNAP_20251218_194959__N9.csv"

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
