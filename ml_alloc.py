# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 19:13:13 2025

@author: mauro

Allocation module for ml2.py

Implements a daily-margin-capped allocator that:
- Takes the already-selected trades from ml2.py (ML Top-K selection).
- Maintains a daily ledger of open margin (open_dt <= D <= close_dt, inclusive).
- Allocates integer lots under a hard daily margin cap (Max Allocation + tolerance).
- Returns trades with P&L, premium, and margin already scaled by the allocated lots.

Trades that cannot receive even 1 lot under the cap are dropped.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd

def apply_max_allocation(
    trades: pd.DataFrame,
    max_allocation: float,
    margin_tolerance: float = 0.0,
    allow_extra_lots: bool = True,
) -> pd.DataFrame:
    """
    Apply daily margin-capped allocation to a set of trades.

    Parameters
    ----------
    trades
        DataFrame with at least:
        - 'strategy_uid'
        - 'open_dt' (datetime64[ns])
        - 'close_dt' (datetime64[ns])
        - 'margin_req' (per-unit margin)
        - 'pnl'
        - 'pnl_R'
        - 'premium'
        Optionally:
        - 'p_pred' (probability used for ranking).

    max_allocation
        Daily margin cap (same units as margin_req).

    margin_tolerance
        Allowed slack on the cap (cap + tolerance) when testing each extra lot.

    allow_extra_lots
        If False:
            - At most 1 lot per trade (baseline-style).
            - No extra-lots round; trades that cannot get even 1 lot are dropped.
        If True:
            - 1 base lot if possible, then multiple passes to add extra lots by rank.
    """
    if trades is None or trades.empty:
        return trades

    if max_allocation is None or max_allocation <= 0:
        print("[ALLOC] max_allocation <= 0 → allocator disabled, returning original trades.")
        return trades

    total_in = len(trades)
    print(f"[ALLOC] Incoming ML-selected trades: {total_in}")

    t = trades.copy()

    # Normalize to date-only for the ledger
    t["open_date"] = pd.to_datetime(t["open_dt"], errors="coerce").dt.normalize()
    t["close_date"] = pd.to_datetime(t["close_dt"], errors="coerce").dt.normalize()

    # Identify invalid rows (for diagnostics)
    invalid_mask = (
        t["open_date"].isna()
        | t["close_date"].isna()
        | t["margin_req"].isna()
    )
    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        print(f"[ALLOC] WARNING: {invalid_count} trades dropped as INVALID "
              f"(NaN open_date/close_date/margin_req).")
        print(t.loc[invalid_mask, ["strategy_uid", "open_dt", "close_dt", "margin_req"]]
              .head(10)
              .to_string(index=False))

    valid_mask = ~invalid_mask
    t = t.loc[valid_mask].copy()
    total_after_valid = len(t)
    print(f"[ALLOC] Trades after validity filter: {total_after_valid} "
          f"(dropped {total_in - total_after_valid} total invalid rows).")

    if t.empty:
        print("[ALLOC] All trades invalid after filter → returning original trades unchanged.")
        return trades

    # Full calendar of dates from first open to last close
    start = t["open_date"].min()
    end = t["close_date"].max()
    all_days = pd.date_range(start=start, end=end, freq="D")

    # Daily margin ledger
    margin_by_day = pd.Series(0.0, index=all_days)

    # Lot counter per trade (integer >= 0)
    t["lots"] = 0

    # Ranking score
    if "p_pred" in t.columns:
        t["_rank_p"] = t["p_pred"].fillna(0.0)
    else:
        t["_rank_p"] = 0.0

    # Group trades by open_date so allocation is done day by day
    grouped_idx = t.groupby("open_date").groups

    def _can_add_lot(idx_row: int) -> bool:
        row = t.loc[idx_row]
        k = float(row["margin_req"])
        if not np.isfinite(k) or k <= 0.0:
            return False

        o = row["open_date"]
        c = row["close_date"]
        d_range = pd.date_range(start=o, end=c, freq="D")

        return bool(
            ((margin_by_day[d_range] + k) <= (max_allocation + margin_tolerance)).all()
        )

    def _apply_lot(idx_row: int) -> None:
        nonlocal margin_by_day

        row = t.loc[idx_row]
        k = float(row["margin_req"])
        o = row["open_date"]
        c = row["close_date"]
        d_range = pd.date_range(start=o, end=c, freq="D")

        margin_by_day[d_range] = margin_by_day[d_range] + k
        t.at[idx_row, "lots"] = t.at[idx_row, "lots"] + 1

    # --- optional diagnostic of live margin path with 1-lot per trade (if you already added it, keep it here) ---
    # (leave this part as you have it, I won't repeat it to avoid confusion)
    # -------------------------------------------------------------------------

    # Main allocation loop: go day by day.
    for day in all_days:
        idxs_today = grouped_idx.get(day, None)
        if idxs_today is None or len(idxs_today) == 0:
            continue

        idx_list = list(idxs_today)
        day_slice = t.loc[idx_list].copy()
        day_slice = day_slice.sort_values(
            by=["_rank_p", "open_dt", "strategy_uid"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        ordered_idx = list(day_slice.index)

        # Pass 1: give at most one base lot per trade (hard cap may block some)
        for idx_row in ordered_idx:
            if _can_add_lot(idx_row):
                _apply_lot(idx_row)

        # Pass 2+: only if extra lots are allowed (ML series).
        if allow_extra_lots:
            something_added = True
            while something_added:
                something_added = False
                for idx_row in ordered_idx:
                    # Never add extra lots to a trade that did not get a base lot.
                    if t.at[idx_row, "lots"] <= 0:
                        continue
                    if _can_add_lot(idx_row):
                        _apply_lot(idx_row)
                        something_added = True

    # Diagnostics: dropped due to cap
    dropped_cap_mask = (t["lots"] == 0) & t["margin_req"].gt(0)
    dropped_cap_count = int(dropped_cap_mask.sum())
    if dropped_cap_count > 0:
        print(f"[ALLOC] WARNING: {dropped_cap_count} trades dropped due to "
              f"margin cap (could not allocate even 1 lot).")
        cols_to_show = ["strategy_uid", "open_dt", "close_dt", "margin_req"]
        if "p_pred" in t.columns:
            cols_to_show.append("p_pred")
        print(
            t.loc[dropped_cap_mask, cols_to_show]
            .head(20)
            .to_string(index=False)
        )

    # Keep only trades with at least 1 lot
    t = t[t["lots"] > 0].copy()
    total_out = len(t)
    print(f"[ALLOC] Trades with at least 1 lot: {total_out} "
          f"(dropped {total_after_valid - total_out} after allocation).")

    if t.empty:
        print("[ALLOC] All trades ended up at 0 lots → returning original trades unchanged.")
        return trades

    # Scale monetary columns by the allocated lots
    for col in ("pnl", "pnl_R", "premium", "margin_req"):
        if col in t.columns:
            t[col] = t[col] * t["lots"].astype(float)

    # Drop helper columns (we keep p_pred if present)
    t = t.drop(
        columns=[c for c in ["open_date", "close_date", "_rank_p", "lots"] if c in t.columns]
    )

    # Preserve original column order as much as possible
    original_cols = list(trades.columns)
    extra_cols = [c for c in t.columns if c not in original_cols]
    cols_final = [c for c in original_cols if c in t.columns] + extra_cols
    t = t[cols_final]

    return t



# def apply_max_allocation(
#     trades: pd.DataFrame,
#     max_allocation: float,
#     margin_tolerance: float = 0.0,
# ) -> pd.DataFrame:
#     """
#     Apply daily margin-capped allocation to the ML-selected trades.

#     Parameters
#     ----------
#     trades
#         DataFrame with at least:
#         - 'strategy_uid'
#         - 'open_dt' (datetime64[ns])
#         - 'close_dt' (datetime64[ns])
#         - 'margin_req' (per-unit margin)
#         - 'pnl'
#         - 'pnl_R'
#         - 'premium'
#         Optionally:
#         - 'p_pred' (probability used for ranking).

#     max_allocation
#         Daily margin cap (same units as margin_req).

#     margin_tolerance
#         Allowed slack on the cap (cap + tolerance) when testing each extra lot.

#     Returns
#     -------
#     DataFrame
#         Trades that actually receive at least 1 lot, with:
#         - pnl, pnl_R, premium, margin_req scaled by allocated lots.
#         - helper columns removed (no 'lots' column in the output).
#         - 'p_pred' preserved if present.
#     """
#     if trades is None or trades.empty:
#         return trades

#     if max_allocation is None or max_allocation <= 0:
#         # Degenerate cap -> keep equal-size behaviour.
#         return trades

#     t = trades.copy()

#     # Normalize to date-only for the ledger
#     t["open_date"] = pd.to_datetime(t["open_dt"], errors="coerce").dt.normalize()
#     t["close_date"] = pd.to_datetime(t["close_dt"], errors="coerce").dt.normalize()

#     # Require valid dates and margin
#     valid_mask = (
#         t["open_date"].notna()
#         & t["close_date"].notna()
#         & t["margin_req"].notna()
#     )
#     t = t.loc[valid_mask].copy()
#     if t.empty:
#         # If nothing valid, fall back to original trades
#         return trades

#     # Full calendar of dates from first open to last close
#     start = t["open_date"].min()
#     end = t["close_date"].max()
#     all_days = pd.date_range(start=start, end=end, freq="D")

#     # Daily margin ledger
#     margin_by_day = pd.Series(0.0, index=all_days)

#     # Lot counter per trade (integer >= 0)
#     t["lots"] = 0

#     # Ranking score
#     if "p_pred" in t.columns:
#         t["_rank_p"] = t["p_pred"].fillna(0.0)
#     else:
#         t["_rank_p"] = 0.0

#     # Group trades by open_date so allocation is done day by day
#     grouped_idx = t.groupby("open_date").groups

#     def _can_add_lot(idx_row: int) -> bool:
#         """
#         Check whether we can add 1 more lot to this trade
#         without exceeding the margin cap on ANY day the trade is open.
#         """
#         row = t.loc[idx_row]
#         k = float(row["margin_req"])
#         if k <= 0:
#             return False

#         o = row["open_date"]
#         c = row["close_date"]
#         d_range = pd.date_range(start=o, end=c, freq="D")

#         # Hard cap: all days must stay <= max_allocation + margin_tolerance
#         return bool(
#             ((margin_by_day[d_range] + k) <= (max_allocation + margin_tolerance)).all()
#         )

#     def _apply_lot(idx_row: int) -> None:
#         """
#         Apply 1 lot to the given trade, updating the ledger.
#         """
#         nonlocal margin_by_day

#         row = t.loc[idx_row]
#         k = float(row["margin_req"])
#         o = row["open_date"]
#         c = row["close_date"]
#         d_range = pd.date_range(start=o, end=c, freq="D")

#         margin_by_day[d_range] = margin_by_day[d_range] + k
#         t.at[idx_row, "lots"] = t.at[idx_row, "lots"] + 1

#     # Main allocation loop: go day by day.
#     for day in all_days:
#         idxs_today = grouped_idx.get(day, None)
#         if idxs_today is None or len(idxs_today) == 0:
#             continue

#         # Deterministic priority:
#         #   1) higher p_pred
#         #   2) earlier open_dt
#         #   3) strategy_uid as tiebreaker
#         idx_list = list(idxs_today)
#         day_slice = t.loc[idx_list].copy()
#         day_slice = day_slice.sort_values(
#             by=["_rank_p", "open_dt", "strategy_uid"],
#             ascending=[False, True, True],
#             kind="mergesort",
#         )
#         ordered_idx = list(day_slice.index)

#         # Pass 1: give at most one base lot per trade (hard cap may block some)
#         for idx_row in ordered_idx:
#             if _can_add_lot(idx_row):
#                 _apply_lot(idx_row)
#             # else: left at 0 lots => dropped under the cap

#         # Pass 2+: keep cycling over trades, adding extra lots where possible
#         # until no more trades can receive an extra lot.
#         something_added = True
#         while something_added:
#             something_added = False
#             for idx_row in ordered_idx:
#                 # Never add extra lots to a trade that did not get a base lot.
#                 if t.at[idx_row, "lots"] <= 0:
#                     continue
#                 if _can_add_lot(idx_row):
#                     _apply_lot(idx_row)
#                     something_added = True

#     # Keep only trades with at least 1 lot
#     t = t[t["lots"] > 0].copy()
#     if t.empty:
#         # Under extreme caps we could wipe everything; in that case,
#         # keep original unscaled trades to avoid returning an empty run.
#         return trades

#     # Scale monetary columns by the allocated lots
#     for col in ("pnl", "pnl_R", "premium", "margin_req"):
#         if col in t.columns:
#             t[col] = t[col] * t["lots"].astype(float)

#     # Drop helper columns (we keep p_pred if present)
#     t = t.drop(
#         columns=[c for c in ["open_date", "close_date", "_rank_p", "lots"] if c in t.columns]
#     )

#     # Try to preserve original column order as much as possible
#     original_cols = list(trades.columns)
#     extra_cols = [c for c in t.columns if c not in original_cols]
#     cols_final = [c for c in original_cols if c in t.columns] + extra_cols
#     t = t[cols_final]

#     return t
