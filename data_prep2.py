# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 18:23:38 2025

@author: mauro

Parent portfolio registry and utility helpers for Phase 3 (CPO-lite).

- Registry JSON: <ml_root>/parents.json
- Parent CSVs:    <ml_root>/parents/<parent_id>.csv

A "parent" is just a wrapper that groups several Phase-2 portfolios
("children") together for CPO-lite.
"""



from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from core import registry

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

ML_ROOT = Path(__file__).resolve().parent

# Registry of parents (JSON)
PARENTS_REGISTRY_PATH = ML_ROOT / "parents.json"

# Folder for parent CSV datasets
PARENTS_DATA_DIR = ML_ROOT / "parents"
PARENTS_DATA_DIR.mkdir(parents=True, exist_ok=True)



# ---------------------------------------------------------------------
# Internal helpers: load/save parents.json
# ---------------------------------------------------------------------

def _load_parents_registry() -> Dict[str, Any]:
    """
    Load parents.json. If missing, return empty structure.
    """
    if not PARENTS_REGISTRY_PATH.exists():
        return {"version": 1, "parents": []}

    with PARENTS_REGISTRY_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Basic sanity
    if "parents" not in data or not isinstance(data["parents"], list):
        data = {"version": 1, "parents": []}
    return data


def _save_parents_registry(data: Dict[str, Any]) -> None:
    """
    Save parents.json (pretty-printed).
    """
    data.setdefault("version", 1)
    data.setdefault("parents", [])
    with PARENTS_REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _parent_csv_path(parent_id: str) -> Path:
    """
    Path to the CSV dataset for a given parent.
    """
    return PARENTS_DATA_DIR / f"{parent_id}.csv"


# ---------------------------------------------------------------------
# Public API used by page4.py
# ---------------------------------------------------------------------

def list_parents() -> List[Dict[str, Any]]:
    """
    Return a list of all parent definitions.
    Each parent dict has at least: id, name, child_ids, csv_path.
    """
    data = _load_parents_registry()
    return data.get("parents", [])


def get_parent(parent_id: str) -> Optional[Dict[str, Any]]:
    """
    Return a single parent by id, or None if not found.
    """
    if not parent_id:
        return None

    data = _load_parents_registry()
    for p in data.get("parents", []):
        if p.get("id") == parent_id:
            return p
    return None


def _generate_parent_id(existing_ids: List[str]) -> str:
    """
    Simple unique id generator for parents.
    """
    base = "parent"
    i = 1
    while f"{base}_{i}" in existing_ids:
        i += 1
    return f"{base}_{i}"


def create_parent(name: str, child_ids: List[str]) -> Dict[str, Any]:
    """
    Create a new parent:

    - Register it in parents.json
    - Build its CSV dataset
    - Return the parent dict
    """
    data = _load_parents_registry()
    parents = data.get("parents", [])

    existing_ids = [p.get("id", "") for p in parents]
    parent_id = _generate_parent_id(existing_ids)

    parent: Dict[str, Any] = {
        "id": parent_id,
        "name": name.strip(),
        "child_ids": list(child_ids),
    }

    # Build CSV snapshot for this parent
    csv_path = build_parent_dataset(parent)
    parent["csv_path"] = str(csv_path)

    parents.append(parent)
    data["parents"] = parents
    _save_parents_registry(data)

    return parent


def update_parent(parent_id: str, name: str, child_ids: List[str]) -> Dict[str, Any]:
    """
    Update an existing parent (name + child_ids) and rebuild its CSV.
    """
    data = _load_parents_registry()
    parents = data.get("parents", [])

    idx = None
    for i, p in enumerate(parents):
        if p.get("id") == parent_id:
            idx = i
            break

    if idx is None:
        raise ValueError(f"Parent id '{parent_id}' not found in parents.json")

    parent = parents[idx]
    parent["name"] = name.strip()
    parent["child_ids"] = list(child_ids)

    # Rebuild CSV snapshot
    csv_path = build_parent_dataset(parent)
    parent["csv_path"] = str(csv_path)

    parents[idx] = parent
    data["parents"] = parents
    _save_parents_registry(data)

    return parent


def delete_parent(parent_id: str, *, delete_csv: bool = True) -> bool:
    """
    Delete a parent from registry (and its CSV file if delete_csv=True).
    Returns True if something was deleted.
    """
    data = _load_parents_registry()
    parents = data.get("parents", [])

    new_parents = []
    deleted = False
    csv_path_to_remove: Optional[Path] = None

    for p in parents:
        if p.get("id") == parent_id:
            deleted = True
            if delete_csv:
                csv_path = p.get("csv_path")
                if csv_path:
                    csv_path_to_remove = Path(csv_path)
        else:
            new_parents.append(p)

    if not deleted:
        return False

    data["parents"] = new_parents
    _save_parents_registry(data)

    if delete_csv and csv_path_to_remove is not None and csv_path_to_remove.exists():
        try:
            csv_path_to_remove.unlink()
        except OSError:
            # Non-fatal: the CSV may already have been removed manually
            pass

    return True


# ---------------------------------------------------------------------
# Core: build parent dataset from child portfolios
# ---------------------------------------------------------------------

def _load_child_portfolio_timeseries(child_portfolio_id: str) -> pd.DataFrame:
    """
    Build a *daily* P&L series for a single child portfolio, using
    the Phase-2 portfolio definition in core.registry.

    Result columns:
        date        (datetime64[ns])
        child_id    (str)
        child_name  (str)
        pnl         (float, $ per day)

    This version uses the actual Phase-1 file headers:
        - 'Date Opened' for the trade date
        - 'P/L'       for dollar P&L
    """
    pf = registry.get_portfolio(child_portfolio_id)
    if pf is None:
        raise ValueError(f"Portfolio '{child_portfolio_id}' not found in registry.")

    child_name = pf.get("name", child_portfolio_id)
    strategy_uids = pf.get("strategy_uids", []) or []

    # Optional: weights map, keyed by strategy_uid. If absent, use 1.0 for all.
    weights_map: Dict[str, float] = pf.get("weights", {}) or {}

    daily_frames: List[pd.DataFrame] = []

    for uid in strategy_uids:
        strat = registry.get_strategy_by_uid(uid)
        if strat is None:
            # Strategy removed / missing; skip
            continue

        file_path_str = strat.get("file_path")
        if not file_path_str:
            continue

        path = Path(file_path_str)
        if not path.exists():
            continue

        df_strat = pd.read_csv(path)

        # HARD-CODED: these are your real column names
        if "Date Opened" not in df_strat.columns or "P/L" not in df_strat.columns:
            raise ValueError(
                f"Strategy file {path} must contain columns 'Date Opened' and 'P/L'."
            )

        # Normalize Date Opened to date-only
        df_strat["Date Opened"] = pd.to_datetime(
            df_strat["Date Opened"], errors="coerce"
        ).dt.normalize()

        df_strat = df_strat.dropna(subset=["Date Opened", "P/L"])
        if df_strat.empty:
            continue

        # Apply portfolio weight for this strategy if present
        w = float(weights_map.get(uid, 1.0))
        df_strat["weighted_pnl"] = df_strat["P/L"].astype(float) * w

        # Keep only what we need for daily aggregation
        daily_frames.append(df_strat[["Date Opened", "weighted_pnl"]])

    if not daily_frames:
        # No strategies / no data: return empty frame with correct columns
        return pd.DataFrame(columns=["date", "child_id", "child_name", "pnl"])

    df_all = pd.concat(daily_frames, ignore_index=True)

    # Aggregate by day
    grouped = (
        df_all
        .groupby("Date Opened", as_index=False)["weighted_pnl"]
        .sum()
        .rename(columns={"Date Opened": "date", "weighted_pnl": "pnl"})
    )

    grouped["child_id"] = child_portfolio_id
    grouped["child_name"] = child_name

    # Ensure column order
    grouped = grouped[["date", "child_id", "child_name", "pnl"]]

    return grouped

def _add_child_control_features(df_parent: pd.DataFrame) -> pd.DataFrame:
    """
    Given a parent dataframe with columns:
        date, child_id, child_name, pnl
    compute per-child rolling control features and return an
    augmented dataframe.

    All rolling stats are computed on pnl shifted by 1 day, so that
    features at date T only use information up to T-1 (no target leakage).
    """
    if df_parent.empty:
        return df_parent

    df = df_parent.copy()
    # Ensure canonical ordering
    df = df.sort_values(["child_id", "date"]).reset_index(drop=True)

    # Pre-create columns filled with NaN
    ctrl_cols = [
        "child_mean_20d",
        "child_std_20d",
        "child_winrate_20d",
        "child_mean_60d",
        "child_std_60d",
        "child_winrate_60d",
        "child_dd_60d",
    ]
    for col in ctrl_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Compute rolling metrics per child
    for cid, grp in df.groupby("child_id"):
        idx = grp.index
        # Use pnl up to yesterday only
        s = grp["pnl"].astype(float)
        s_shift = s.shift(1)

        # 20-day window
        mean_20 = s_shift.rolling(window=20, min_periods=5).mean()
        std_20 = s_shift.rolling(window=20, min_periods=5).std()
        win_20 = (s_shift > 0).rolling(window=20, min_periods=5).mean()

        # 60-day window
        mean_60 = s_shift.rolling(window=60, min_periods=10).mean()
        std_60 = s_shift.rolling(window=60, min_periods=10).std()
        win_60 = (s_shift > 0).rolling(window=60, min_periods=10).mean()

        # 60-day rolling max drawdown on cumulative pnl (up to yesterday)
        eq = s_shift.fillna(0.0).cumsum()
        roll_max = eq.rolling(window=60, min_periods=10).max()
        dd = eq - roll_max  # negative or zero
        dd_60 = dd.rolling(window=60, min_periods=10).min()
        # Store magnitude of worst drawdown (positive number)
        dd_60_mag = dd_60.abs()

        df.loc[idx, "child_mean_20d"] = mean_20.values
        df.loc[idx, "child_std_20d"] = std_20.values
        df.loc[idx, "child_winrate_20d"] = win_20.values

        df.loc[idx, "child_mean_60d"] = mean_60.values
        df.loc[idx, "child_std_60d"] = std_60.values
        df.loc[idx, "child_winrate_60d"] = win_60.values

        df.loc[idx, "child_dd_60d"] = dd_60_mag.values

    return df


def build_parent_dataset(parent: Dict[str, Any]) -> Path:
    """
    Build the CSV dataset for a parent:

    - For each child portfolio id in parent["child_ids"]:
        * Build its daily P&L series
    - Stack all children
    - Add per-child rolling control features
    - Save to ml/parents/<parent_id>.csv

    Result CSV schema (minimum):

        date, child_id, child_name, pnl,
        child_mean_20d, child_std_20d, child_winrate_20d,
        child_mean_60d, child_std_60d, child_winrate_60d,
        child_dd_60d
    """
    child_ids: List[str] = parent.get("child_ids", []) or []

    all_frames: List[pd.DataFrame] = []
    for cid in child_ids:
        df_child = _load_child_portfolio_timeseries(cid)
        if not df_child.empty:
            all_frames.append(df_child)

    if all_frames:
        df_parent = pd.concat(all_frames, ignore_index=True)
        df_parent = df_parent.sort_values(["date", "child_id"]).reset_index(drop=True)

        # Add per-child control / descriptor features (rolling stats on pnl)
        df_parent = _add_child_control_features(df_parent)
    else:
        # Empty but with correct columns
        df_parent = pd.DataFrame(
            columns=[
                "date",
                "child_id",
                "child_name",
                "pnl",
                "child_mean_20d",
                "child_std_20d",
                "child_winrate_20d",
                "child_mean_60d",
                "child_std_60d",
                "child_winrate_60d",
                "child_dd_60d",
            ]
        )

    csv_path = _parent_csv_path(parent["id"])
    # WRITE THE FILE (always, even if empty)
    df_parent.to_csv(csv_path, index=False)

    # HARD LOG so you see it in the console
    print(
        f"[data_prep2] build_parent_dataset: wrote {df_parent.shape[0]} rows "
        f"for parent '{parent['id']}' to: {csv_path}"
    )

    return csv_path
