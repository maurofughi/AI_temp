# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 20:31:56 2025

@author: mauro


Portfolio26 - Phase 3 ML
ml4.py

Daily Portfolio CPO-Lite Walk-Forward Analysis (FWA) using LightGBM  .
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

from ml.ml_alloc import apply_max_allocation

import os


try:
    from lightgbm import LGBMRegressor
except Exception as e:
    LGBMRegressor = None
    _LGBM_IMPORT_ERROR = e
    
try:
    import shap
except ImportError:
    shap = None


    
# ---------------------------------------------------------------------
# Global RNG seed for feature randomization (price, VIX, gap, etc.)
# ---------------------------------------------------------------------
RANDOM_SEED = 113723324  # change this if you want different random paths


# -------------------------
# Parameters
# -------------------------
@dataclass
class RunParamsDaily:
    dataset_csv_path: str
    # optional path to extra daily feature file; if None, a default
    # relative path is derived from dataset_csv_path
    extra_features_csv_path: Optional[str] = None

    # optional overall bounds (date-only strings "YYYY-MM-DD")
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # IS window length (months) - unchanged conceptually
    is_months: int = 2

    # OoS length (TRADING DAYS)
    oos_days: int = 1

    anchored_type: str = "U"   # "U" (Unanchored) or "A" (Anchored)
    
    # ------------------ SELECTION MODE (NEW) ------------------
    # "top_k":   select Top K strategies per day (current behaviour)
    # "bottom_k": drop Bottom K strategies per day (avoid worst)
    # "bottom_p":  drop all strategies whose ML rank is in the worst p% (global percentile)
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
    max_allocation: float = 12000.0      # example default; tune as needed
    allocation_tolerance: float = 0.0    # slack on the cap
    

TARGET_COL = "pnl"
PNL_COL = "pnl"
PNLR_COL = "pnl_R"
# Still needed by PCR/participation metrics. At parent level this is just
# a dummy column = 0.0, but we keep the constant so downstream code works.
PREMIUM_COL = "premium"



# -------------------------
# Utils: date parsing
# -------------------------
def _to_ts_date(d: str) -> pd.Timestamp:
    # normalize to date-only midnight
    return pd.Timestamp(d).normalize()


def _date_floor_from_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _compute_cycles_daily(
    params: RunParamsDaily,
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    open_dates: pd.Series,  # unique, normalized, sorted
) -> pd.DataFrame:
    """
    Daily cycles (decision calendar = trading days from OPEN dates):

    - OoS is defined by TRADING-DAY blocks on OPEN dates.
    - Cycle step is ALWAYS 1 trading day (not a user param).
    - IS_end = day before OoS_start (calendar day), same convention as ml2 weekly.
    - IS_start:
        Anchored: fixed at overall start
        Unanchored: IS_start = IS_end - is_months + 1 day (month-offset)
    """
    if params.oos_days < 1:
        raise ValueError("Invalid parameters: oos_days must be >= 1.")

    # Overall bounds
    start = _to_ts_date(params.start_date) if params.start_date else data_start.normalize()
    end = _to_ts_date(params.end_date) if params.end_date else data_end.normalize()

    start = max(start, data_start.normalize())
    end = min(end, data_end.normalize())

    d = pd.Series(pd.to_datetime(open_dates, errors="coerce")).dropna().dt.normalize()
    d = d.drop_duplicates().sort_values().reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(columns=["cycle", "IS_from", "IS_to", "OoS_from", "OoS_to"])

    # We need IS_to to have at least is_months of history from start
    # earliest_is_end is the earliest allowed IS_to
    earliest_is_end = (start + pd.DateOffset(months=params.is_months)) - pd.Timedelta(days=1)
    first_oos_start_min = earliest_is_end + pd.Timedelta(days=1)

    # Candidate OoS starts must be trading days >= first_oos_start_min
    candidate_idx = d.index[d >= first_oos_start_min]
    if len(candidate_idx) == 0:
        return pd.DataFrame(columns=["cycle", "IS_from", "IS_to", "OoS_from", "OoS_to"])

    i0 = int(candidate_idx[0])

    cycles = []
    anchored_is_from = start

    i = i0
    safety = 0
    while i < len(d):
        oos_from = pd.Timestamp(d.iloc[i]).normalize()
        if oos_from > end:
            break

        j = i + int(params.oos_days) - 1
        if j >= len(d):
            break

        oos_to = pd.Timestamp(d.iloc[j]).normalize()
        if oos_to > end:
            break

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

        # advance by 1 trading day
        i += 1
        safety += 1
        if safety > 50000:
            raise RuntimeError("Too many cycles generated; check daily parameters/date range.")

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
    DAILY CPO-lite OoS prediction panel.

    Row = (OoS day, child portfolio).

    We deliberately *do not* carry over trade-level junk (open_minute, premium,
    margin_req) here. Those columns still exist in the pipeline as placeholders
    for metrics (e.g. PCR via PREMIUM_COL) but they are NOT ML features anymore.

    This function only:

      - Cross-joins OoS days × candidate child IDs.
      - Supplies:
          * open_dt / open_date
          * strategy_uid (child portfolio id)
          * dow (day-of-week)
      - Market-level features (SPX, VIX, gap, opening_price, opening_vix) are
        injected later in run_fwa_daily via:
          - merge with feat_daily on open_date
          - daily CPO block that fills gap / opening_price / opening_vix
    """
    if not oos_days or not strategy_uids:
        return pd.DataFrame()

    # Normalise inputs
    oos_days_norm = [pd.Timestamp(d).normalize() for d in oos_days]
    strategy_uids_norm = [str(s) for s in strategy_uids]

    rows = []
    # Deterministic ordering: days then child portfolios
    for d in sorted(oos_days_norm):
        for uid in sorted(strategy_uids_norm):
            rows.append(
                {
                    "open_dt": d,
                    "open_date": d,
                    "strategy_uid": uid,
                }
            )

    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel

    # Day-of-week for the panel (ML feature)
    panel["dow"] = panel["open_dt"].dt.weekday

    # We DO NOT set opening_price / opening_vix / gap here.
    # They are filled later in run_fwa_daily with:
    #   1) real daily data (gap from OoS pnl file, SPX from feat_daily,
    #      VIX_OPEN_DAY from P26_vix_open.csv);
    #   2) IS-based fallback via _estimate_market_features_from_is for any
    #      remaining NaNs.
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
# Core DAILY CPO-lite FWA run (portfolio-level)
# -------------------------
def run_fwa_daily(params: RunParamsDaily) -> Dict[str, Any]:
    """
    Daily CPO-lite at portfolio level.

    - Dataset: parent file (one row per day per child portfolio).
      Required columns: 'date', 'child_id', 'pnl'.
    - Features: market / state features from P26_extra_features1.csv
      (plus child_idx), NOT trade-level stuff.

    Selection logic:
      - Baseline: per-cycle static "best IS child" (held every OoS day).
      - Random baseline: random child per cycle.
      - ML: Top-K child portfolios per day by LGBM regression on TARGET_COL.
    """
    if LGBMRegressor is None:
        raise ImportError(f"LightGBM is not available: {_LGBM_IMPORT_ERROR}")

    # Global RNG used by baselines and model seeds
    np.random.seed(int(RANDOM_SEED))
    
    print(">>> DEBUG: running ml4.run_fwa_daily with dataset:", params.dataset_csv_path)


    # Default LGBM parameters if none provided
    if params.lgbm_params is None:
        params.lgbm_params = dict(
            n_estimators=280,
            learning_rate=0.05,
            num_leaves=25,
            max_depth=-1,
            min_data_in_leaf=50,
            subsample=0.8,
            colsample_bytree=0.8,
            lambda_l2=25,
            min_gain_to_split=0.05,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
            bagging_seed=RANDOM_SEED,
            feature_fraction_seed=RANDOM_SEED,
            data_random_seed=RANDOM_SEED,
            force_col_wise=True,
            deterministic=True,
        )

    # ------------------------------------------------------------------
    # Load parent-portfolio dataset (one row per day per child portfolio)
    # Required columns from data_prep2: date, child_id, pnl
    # ------------------------------------------------------------------
    df_raw = pd.read_csv(params.dataset_csv_path)

    if "date" not in df_raw.columns:
        raise ValueError("Parent file must have a 'date' column.")
    if "child_id" not in df_raw.columns:
        raise ValueError("Parent file must have a 'child_id' column.")
    if "pnl" not in df_raw.columns:
        raise ValueError("Parent file must have a 'pnl' column.")

    # Map date -> open_dt / close_dt (same at portfolio level, daily resolution)
    df_raw["open_dt"] = pd.to_datetime(df_raw["date"], errors="coerce")
    df_raw["close_dt"] = df_raw["open_dt"]
    df_raw["open_date"] = df_raw["open_dt"].dt.normalize()

    # Child portfolio identifiers -> strategy_uid for downstream code
    df_raw["strategy_uid"] = df_raw["child_id"].astype(str)
    if "child_name" in df_raw.columns:
        df_raw["strategy_name"] = df_raw["child_name"].astype(str)
    else:
        df_raw["strategy_name"] = df_raw["strategy_uid"]

    # Target and P&L columns
    df_raw[PNL_COL] = df_raw["pnl"].astype(float)
    if PNLR_COL not in df_raw.columns:
        # At parent level treat pnl_R = pnl (or adjust if you prefer)
        df_raw[PNLR_COL] = df_raw[PNL_COL]

    # Placeholder columns needed by allocation/metrics functions
    if PREMIUM_COL not in df_raw.columns:
        df_raw[PREMIUM_COL] = 0.0
    if "margin_req" not in df_raw.columns:
        df_raw["margin_req"] = 0.0

    # Clean basic structure
    df = df_raw.dropna(subset=["open_dt", "close_dt", PNL_COL, PNLR_COL]).copy()
    df = df.sort_values("open_dt").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Child index feature (this is the portfolio ID signal)
    # ------------------------------------------------------------------
    uid_values = sorted(df["strategy_uid"].unique())
    uid_to_idx = {uid: i for i, uid in enumerate(uid_values)}
    df["child_idx"] = df["strategy_uid"].map(uid_to_idx).astype("int32")

    # ------------------------------------------------------------------
    # Load and derive extra daily features (SPX/VIX etc.)
    # ------------------------------------------------------------------
    extra_path = params.extra_features_csv_path
    if extra_path is None:
        dataset_dir = os.path.dirname(params.dataset_csv_path)
        ml_root = os.path.dirname(dataset_dir)
        extra_path = os.path.join(ml_root, "features", "P26_extra_features1.csv")

    print(f"Loading extra features: {extra_path}")
    if not os.path.exists(extra_path):
        raise FileNotFoundError(f"Extra features file not found: {extra_path}")

    feat_daily = pd.read_csv(extra_path)

    if "tradedate" not in feat_daily.columns:
        raise ValueError("Extra features file must have 'tradedate' column.")

    # Normalize tradedate to open_date (date-only) for joining
    feat_daily["open_date"] = pd.to_datetime(
        feat_daily["tradedate"], errors="coerce"
    ).dt.normalize()
    feat_daily = feat_daily.drop(columns=["tradedate"])
    feat_daily = feat_daily.sort_values("open_date").reset_index(drop=True)

    # Sanity: SPX & VIX present for derived features (but they WILL NOT be used as features)
    for _col in ["SPX", "VIX"]:
        if _col not in feat_daily.columns:
            raise ValueError(f"Extra features file must contain '{_col}' column for derived features.")

    # --- SPX-based derived features (using close SPX; not fed directly to model) ---
    spx = feat_daily["SPX"].astype(float)
    feat_daily["spx_ret_5d"] = spx.pct_change(5)
    feat_daily["spx_ret_20d"] = spx.pct_change(20)
    spx_tr = spx.diff().abs()
    feat_daily["spx_atr14"] = spx_tr.rolling(14, min_periods=5).mean()
    feat_daily["spx_sma_20"] = spx.rolling(20, min_periods=5).mean()
    feat_daily["spx_sma_50"] = spx.rolling(50, min_periods=10).mean()

    # --- VIX-based derived features (using close VIX; not fed directly to model) ---
    vix = feat_daily["VIX"].astype(float)
    feat_daily["vix_mean_5d"] = vix.rolling(5, min_periods=3).mean()
    feat_daily["vix_mean_20d"] = vix.rolling(20, min_periods=5).mean()

    # At this point feat_daily should also already contain:
    #   opening_price, opening_vix, gap
    # created by you in P26_extra_features1.csv.
    # We will feed those, not SPX/VIX closes, to avoid leakage.

    # Identify extra feature columns = everything except join key AND SPX/VIX
    extra_feature_cols = [
        c for c in feat_daily.columns
        if c not in ("open_date", "SPX", "VIX")
    ]

    print("Extra feature columns used for ML:", extra_feature_cols)

    # Merge extra features into the main dataset on open_date
    df = df.merge(feat_daily, on="open_date", how="left")

    # Final feature set LightGBM will see
    feature_cols_loc = extra_feature_cols + ["child_idx"]

    # Basic date bounds
    data_start = df["open_dt"].min().normalize()
    data_end = df["open_dt"].max().normalize()

    # Build trading day table from open dates (decision calendar)
    open_dates = df["open_dt"].dt.normalize().dropna().drop_duplicates().sort_values()
    cycles_df = _compute_cycles_daily(params, data_start, data_end, open_dates)

    DEBUG_CYCLES = {55, 255, 455}

    print("\nRUN PARAMS (DAILY CPO-LITE, portfolio-level):")
    print(f" dataset: {params.dataset_csv_path}")
    print(
        f" anchored_type: {params.anchored_type} "
        f"IS={params.is_months} months OoS={params.oos_days} days step=1 day"
    )
    print(f" top_k_per_day: {params.top_k_per_day}")
    print(
        f" allocation_mode: {params.allocation_mode} "
        f"max_allocation={params.max_allocation} "
        f"tolerance={params.allocation_tolerance}"
    )
    print(f" features (X): {feature_cols_loc}")
    print(f" target (y): {TARGET_COL} (regressed pnl)")

    oos_selected_rows: List[pd.DataFrame] = []
    baseline_static_rows: List[pd.DataFrame] = []
    baseline_random_rows: List[pd.DataFrame] = []
    cycle_summaries: List[Dict[str, Any]] = []

    if cycles_df.empty:
        base_all = pd.DataFrame()
        sel_all = pd.DataFrame()
        rand_all = pd.DataFrame()
        return {
            "params": params,
            "cycles": cycles_df,
            "cycle_summaries": pd.DataFrame(),
            "baseline_trades": base_all,
            "selected_trades": sel_all,
            "random_baseline_trades": rand_all,
            "baseline_metrics": _compute_metrics(base_all, params.initial_equity),
            "ml_metrics": _compute_metrics(sel_all, params.initial_equity),
            "random_baseline_metrics": _compute_metrics(rand_all, params.initial_equity),
        }

    # ------------- MAIN FWA LOOP OVER CYCLES -------------
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

        # Skip cycle if no IS/OoS rows
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

        # -------------- BASELINES (per cycle) --------------
        # Candidate portfolios in IS
        cand_uids = df_is["strategy_uid"].dropna().unique()

        # All OoS days for this cycle (decision calendar)
        oos_dates_cycle = (
            df_oos_actual["open_dt"].dt.normalize()
            .drop_duplicates()
            .sort_values()
        )

        if len(cand_uids) > 0 and len(oos_dates_cycle) > 0:
            # ---- Static-best baseline: child with highest IS pnl_R ----
            is_perf = df_is.groupby("strategy_uid")[PNLR_COL].sum()
            best_uid = is_perf.idxmax()

            base_static_cycle = pd.DataFrame({"open_date": oos_dates_cycle})
            base_static_cycle["strategy_uid"] = str(best_uid)
            base_static_cycle["open_dt"] = base_static_cycle["open_date"]
            base_static_cycle["close_dt"] = base_static_cycle["open_date"]

            base_actual = df_oos_actual[
                df_oos_actual["strategy_uid"] == best_uid
            ].copy()
            base_actual["open_date"] = base_actual["open_dt"].dt.normalize()

            agg_cols = [PNL_COL, PNLR_COL, PREMIUM_COL, "margin_req"]
            base_actual = (
                base_actual
                .groupby("open_date", as_index=False)[agg_cols]
                .sum()
            )

            base_static_cycle = base_static_cycle.merge(
                base_actual, on="open_date", how="left"
            )
            for col in agg_cols:
                if col not in base_static_cycle.columns:
                    base_static_cycle[col] = 0.0
                else:
                    base_static_cycle[col] = base_static_cycle[col].fillna(0.0)

            baseline_static_rows.append(
                base_static_cycle[
                    ["strategy_uid", "open_dt", "close_dt", PNL_COL, PNLR_COL, PREMIUM_COL, "margin_req"]
                ].copy()
            )

            # ---- Random baseline: pick one candidate uniformly at random for this cycle ----
            rand_uid = np.random.choice(cand_uids)

            base_rand_cycle = pd.DataFrame({"open_date": oos_dates_cycle})
            base_rand_cycle["strategy_uid"] = str(rand_uid)
            base_rand_cycle["open_dt"] = base_rand_cycle["open_date"]
            base_rand_cycle["close_dt"] = base_rand_cycle["open_date"]

            rand_actual = df_oos_actual[
                df_oos_actual["strategy_uid"] == rand_uid
            ].copy()
            rand_actual["open_date"] = rand_actual["open_dt"].dt.normalize()
            rand_actual = (
                rand_actual
                .groupby("open_date", as_index=False)[agg_cols]
                .sum()
            )

            base_rand_cycle = base_rand_cycle.merge(
                rand_actual, on="open_date", how="left"
            )
            for col in agg_cols:
                if col not in base_rand_cycle.columns:
                    base_rand_cycle[col] = 0.0
                else:
                    base_rand_cycle[col] = base_rand_cycle[col].fillna(0.0)

            baseline_random_rows.append(
                base_rand_cycle[
                    ["strategy_uid", "open_dt", "close_dt", PNL_COL, PNLR_COL, PREMIUM_COL, "margin_req"]
                ].copy()
            )

        # -------------- MODEL TRAINING (IS) --------------
        X_is = df_is[feature_cols_loc]
        y_is = df_is[TARGET_COL].astype(float)

        # DEBUG: show exactly what LGBM sees on IS for selected cycles
        if c in DEBUG_CYCLES:
            print(f"\n===== DEBUG CYCLE {c} – IS (X_is, y_is) =====")
            print("X_is columns:", list(X_is.columns))
            print("X_is dtypes:\n", X_is.dtypes)
            print("X_is head:\n", X_is.head(10).to_string(index=False))
            print("y_is head:\n", y_is.head(10).to_string(index=False))

        model = LGBMRegressor(**params.lgbm_params)
        model.fit(X_is, y_is)

        # -------------- BUILD OoS PREDICTION PANEL --------------
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

        # Attach daily features and child_idx
        pred_panel = pred_panel.merge(feat_daily, on="open_date", how="left")
        pred_panel["child_idx"] = pred_panel["strategy_uid"].map(uid_to_idx).astype("int32")

        X_oos = pred_panel[feature_cols_loc]

        # DEBUG: show exactly what LGBM sees on OoS for selected cycles
        if c in DEBUG_CYCLES:
            print(f"\n===== DEBUG CYCLE {c} – OoS (X_oos) =====")
            print("X_oos columns:", list(X_oos.columns))
            print("X_oos dtypes:\n", X_oos.dtypes)
            print("X_oos head:\n", X_oos.head(10).to_string(index=False))

        # -------------- PREDICTION & RANKING --------------
        y_hat = model.predict(X_oos)
        pred_panel["y_hat_raw"] = y_hat
        
        
        # --- DIAGNOSTIC: how much variation in predictions across children? ---
        if c in DEBUG_CYCLES:
            print(f"\n===== DEBUG CYCLE {c} – y_hat variation by child_idx =====")
            tmp = pred_panel.copy()
            # For each child, average prediction over this cycle's OoS days
            avg_pred_by_child = tmp.groupby("child_idx")["y_hat_raw"].mean().sort_values()
            print("Average y_hat_raw by child_idx in this cycle:\n", avg_pred_by_child.to_string())
        
            # Compare to actual mean pnl_R in IS for same children
            avg_is_pnl_by_child = df_is.groupby("child_idx")[PNLR_COL].mean().sort_values()
            print("\nAverage IS pnl_R by child_idx:\n", avg_is_pnl_by_child.to_string())


        n_oos = len(y_hat)
        ml_rank = pd.Series(y_hat).rank(method="average", ascending=True) / n_oos
        pred_panel["p_pred"] = ml_rank.values

        pred_panel["open_date"] = pred_panel["open_date"].dt.normalize()
        df_oos_actual["open_date"] = df_oos_actual["open_dt"].dt.normalize()

        sel_mode = (params.selection_mode or "top_k").lower()
        if sel_mode not in ("top_k",):
            raise ValueError(
                f"ml4.py currently supports selection_mode='top_k' only, got {sel_mode!r}"
            )

        k = int(params.top_k_per_day)
        if k <= 0:
            selected = df_oos_actual.copy()
        else:
            # Top-K child portfolios per day by p_pred
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

        core_cols = [
            "strategy_uid",
            "open_dt",
            "close_dt",
            PNL_COL,
            PNLR_COL,
            PREMIUM_COL,
            "margin_req",
        ]
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
                f"TopK_portfolios/day={params.top_k_per_day} "
                f"Selected_rows={len(selected_core)} pnl_R(sum)={pnlR_sum:.4f} "
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

    # ------------------------------------------------------------------
    # Stitch baselines + ML selected
    # ------------------------------------------------------------------
    base_all = (
        pd.concat(baseline_static_rows, ignore_index=True)
        if baseline_static_rows
        else pd.DataFrame()
    )
    sel_all = (
        pd.concat(oos_selected_rows, ignore_index=True)
        if oos_selected_rows
        else pd.DataFrame()
    )
    rand_all = (
        pd.concat(baseline_random_rows, ignore_index=True)
        if baseline_random_rows
        else pd.DataFrame()
    )

    # Allocation overlay if needed
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
        if not rand_all.empty:
            rand_all = apply_max_allocation(
                trades=rand_all,
                max_allocation=float(params.max_allocation),
                margin_tolerance=float(params.allocation_tolerance),
                allow_extra_lots=False,
            )

    # Sort after allocation
    for _df in (base_all, sel_all, rand_all):
        if not _df.empty:
            _df.sort_values(["close_dt", "open_dt", "strategy_uid"], inplace=True)
            _df.reset_index(drop=True, inplace=True)

    # Metrics & curves
    baseline_metrics = _compute_metrics(
        base_all, initial_equity=float(params.initial_equity)
    )
    ml_metrics = _compute_metrics(
        sel_all, initial_equity=float(params.initial_equity)
    )
    random_baseline_metrics = _compute_metrics(
        rand_all, initial_equity=float(params.initial_equity)
    )

    baseline_curve = _build_daily_series_with_exposure(
        base_all, float(params.initial_equity)
    )
    ml_curve = _build_daily_series_with_exposure(
        sel_all, float(params.initial_equity)
    )
    random_baseline_curve = _build_daily_series_with_exposure(
        rand_all, float(params.initial_equity)
    )

    baseline_nominal = int(df["strategy_uid"].nunique())
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
    random_baseline_extra = _compute_participation_metrics(
        trades=rand_all,
        initial_equity=float(params.initial_equity),
        nominal_units=baseline_nominal,
    )

    # Prefer $ P&L if available
    base_total_pnl = baseline_metrics.get(
        "total_pnl", baseline_metrics.get("total_pnlR", 0.0)
    )
    ml_total_pnl = ml_metrics.get(
        "total_pnl", ml_metrics.get("total_pnlR", 0.0)
    )
    rand_total_pnl = random_baseline_metrics.get(
        "total_pnl", random_baseline_metrics.get("total_pnlR", 0.0)
    )

    print("\nFINAL (stitched by close_dt):")
    print(f" BASELINE trades total: {baseline_metrics['trades']}")
    print(f" BASELINE total pnl: {base_total_pnl:.2f}")
    print(
        f" BASELINE max DD $: {baseline_metrics['max_dd_$']:.2f} "
        f"max DD %: {baseline_metrics['max_dd_%']:.2%}"
    )
    print(f" ML trades total: {ml_metrics['trades']}")
    print(f" ML total pnl: {ml_total_pnl:.2f}")
    print(
        f" ML max DD $: {ml_metrics['max_dd_$']:.2f} "
        f"max DD %: {ml_metrics['max_dd_%']:.2%}"
    )
    print(f" RANDOM BASELINE trades total: {random_baseline_metrics['trades']}")
    print(f" RANDOM BASELINE total pnl: {rand_total_pnl:.2f}")
    print(
        f" RANDOM BASELINE max DD $: {random_baseline_metrics['max_dd_$']:.2f} "
        f"max DD %: {random_baseline_metrics['max_dd_%']:.2%}"
    )

    return {
        "params": params,
        "cycles": cycles_df,
        "cycle_summaries": pd.DataFrame(cycle_summaries),
        # Static best-per-cycle baseline
        "baseline_trades": base_all,
        "baseline_metrics": baseline_metrics,
        "baseline_extra_metrics": baseline_extra,
        "baseline_curve": baseline_curve,
        # ML selection
        "selected_trades": sel_all,
        "ml_metrics": ml_metrics,
        "ml_extra_metrics": ml_extra,
        "ml_curve": ml_curve,
        # Random baseline (diagnostic)
        "random_baseline_trades": rand_all,
        "random_baseline_metrics": random_baseline_metrics,
        "random_baseline_extra_metrics": random_baseline_extra,
        "random_baseline_curve": random_baseline_curve,
    }



# -------------------------------------------------
# Standalone test harness for ml4.py (Spyder usage)
# -------------------------------------------------
if __name__ == "__main__":
    # Point this to one of your parent files created by data_prep2.py
    # Example – adjust to a real file on your machine:
    DATASET = r"C:\Users\mauro\MAURO\Spyder\Portfolio26\ml\parents\parent_1.csv"

    # If you want to override the extra features file, set this; otherwise leave None
    EXTRA_FEAT = None  # or r"C:\Users\mauro\MAURO\Spyder\Portfolio26\ml\features\P26_extra_features1.csv"

    params = RunParamsDaily(
        dataset_csv_path=DATASET,
        extra_features_csv_path=EXTRA_FEAT,
        # You can tweak these if needed; defaults are same as ml3
        # start_date="2022-05-01",
        # end_date="2024-12-31",
        is_months=2,
        oos_days=1,
        anchored_type="U",      # "U" or "A"
        selection_mode="top_k", # or "bottom_k", "bottom_p" if you want
        top_k_per_day=1,
        verbose_cycles=False,
        initial_equity=100000.0,
        allocation_mode="equal",
        max_allocation=12000.0,
        allocation_tolerance=0.0,
    )

    result = run_fwa_daily(params)

    # Quick sanity checks
    # cycles = result.get("cycle_summaries")
    # print("\n=== Cycle summaries (head) ===")
    # if cycles is not None and not cycles.empty:
    #     print(cycles.head(10).to_string(index=False))
    # else:
    #     print("<no cycles>")

    print("\n=== Baseline metrics ===")
    print(result.get("baseline_metrics"))

    print("\n=== ML metrics ===")
    print(result.get("ml_metrics"))
    
    print("\n# baseline_trades:", len(result["baseline_trades"]))
    print("# selected_trades:", len(result["selected_trades"]))
    print("# random_baseline_trades:", len(result["random_baseline_trades"]))

