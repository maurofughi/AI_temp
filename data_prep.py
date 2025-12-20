# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 13:18:43 2025

@author: mauro

Portfolio26 - Phase 3 ML
data_prep.py

Builds the consolidated ML panel dataset from the currently selected strategies,
and writes an immutable snapshot entry into ml_datasets.json.

Design goals:
- No manual pipeline steps.
- Supports unsaved ad-hoc selection (does NOT require a saved portfolio).
- Avoids leakage by only writing entry-known features + targets to the final dataset.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

# Reuse your existing canonical paths / UID logic
from core.registry import (
    derive_uid_from_filepath,
    get_internal_strategy_path,
    INTERNAL_STRATEGY_DIR,
)


# -----------------------------
# Config (local to this module)
# -----------------------------
ML_DIR = Path(__file__).parent
DATASETS_DIR = ML_DIR / "datasets"
INDEX_FILE = ML_DIR / "ml_datasets.json"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Column policy (v1)
# -----------------------------
# Required raw columns from OO logs (minimum to build v1 dataset)
REQ_COLS = [
    "Date Opened",
    "Time Opened",
    "Date Closed",
    "Time Closed",
    "Opening Price",
    "Premium",
    "P/L",
    "No. of Contracts",
    "Margin Req.",
    "Opening VIX",
    "Gap",
]


# Final dataset columns (keys + features + targets)
FINAL_COLS = [
    # keys
    "open_dt",
    "open_date",
    "close_dt",
    "close_date",
    "strategy_uid",

    # features (entry-known)
    "dow",
    "open_minute",
    "opening_price",
    "premium",
    "margin_req",
    "opening_vix",
    "gap",

    # targets
    "pnl",
    "contracts",
    "risk_unit",
    "pnl_R",
    "label",
]



def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_index() -> Dict[str, Any]:
    """
    Load ml_datasets.json, create if missing.
    """
    if not INDEX_FILE.exists():
        data = {"version": 1, "datasets": []}
        _save_index(data)
        return data

    try:
        with INDEX_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": 1, "datasets": []}
        if "datasets" not in data or not isinstance(data["datasets"], list):
            data["datasets"] = []
        if "version" not in data:
            data["version"] = 1
        return data
    except Exception:
        # Do not overwrite on read error
        return {"version": 1, "datasets": []}


def _save_index(data: Dict[str, Any]) -> None:
    tmp = INDEX_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
    tmp.replace(INDEX_FILE)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_strategy_csv_path(strategy_id_or_path: str) -> Tuple[str, Path, str]:
    """
    Input is what your app uses as strategy_id in p1-strategy-checklist:
    - Folder-loaded: full path to CSV
    - Upload-loaded: full path under core/dataupl
    - Saved: may still be original path; if missing on disk, fall back to INTERNAL_STRATEGY_DIR/uid.csv

    Returns:
      (uid, resolved_path, resolution_mode)
    """
    uid = derive_uid_from_filepath(strategy_id_or_path)

    p = Path(strategy_id_or_path)
    if p.exists():
        return uid, p, "as_is"

    # fallback: internal saved copy
    internal = get_internal_strategy_path(uid)
    if internal.exists():
        return uid, internal, "internal_fallback"

    # last resort: try INTERNAL_STRATEGY_DIR explicitly (same as get_internal_strategy_path, but keep clear)
    internal2 = Path(INTERNAL_STRATEGY_DIR) / f"{uid}.csv"
    if internal2.exists():
        return uid, internal2, "internal_dir_fallback"

    raise FileNotFoundError(
        f"Strategy CSV not found. strategy_id='{strategy_id_or_path}', "
        f"uid='{uid}', tried internal='{internal}'."
    )


def _parse_open_dt(df: pd.DataFrame) -> pd.Series:
    """
    Parse open datetime robustly.
    """
    s = df["Date Opened"].astype(str).str.strip() + " " + df["Time Opened"].astype(str).str.strip()

    dt = pd.to_datetime(s, errors="coerce", dayfirst=False)

    # If too many NaT, try dayfirst=True (some exports are dd/mm/yyyy)
    nat_ratio = dt.isna().mean()
    if nat_ratio > 0.10:
        dt2 = pd.to_datetime(s, errors="coerce", dayfirst=True)
        # keep whichever yields fewer NaT
        if dt2.isna().mean() < nat_ratio:
            dt = dt2

    return dt


def build_ml_panel_dataset(
    selected_strategy_ids: List[str],
    weights_store: Optional[Dict[str, Any]] = None,
    source_portfolio_id: Optional[str] = None,
    source_portfolio_name: Optional[str] = None,
    save_to_disk: bool = True,
) -> Dict[str, Any]:
    """
    Build the ML panel dataset and create a snapshot entry.

    Parameters
    ----------
    selected_strategy_ids : list[str]
        Typically p1-strategy-checklist.value (strategy ids are file paths in your app).
    weights_store : dict
        Typically p2-weights-store.data: {uid: {"factor": float}, ...}
        Used for snapshot metadata only (not written into dataset file in v1).
    source_portfolio_id/name : optional
        For traceability only (if a saved portfolio was loaded).
    save_to_disk : bool
        If True, save dataset CSV under ml/datasets/

    Returns
    -------
    dict with keys:
      - snapshot_id
      - dataset_path (or None)
      - n_rows
      - min_open_dt / max_open_dt
      - selected_uids
      - errors (list)
    """
    weights_store = weights_store or {}

    errors: List[str] = []
    rows_all: List[pd.DataFrame] = []

    # Snapshot metadata collectors
    selected_uids: List[str] = []
    strategy_sources: Dict[str, Dict[str, str]] = {}
    strategy_factors: Dict[str, float] = {}

    # Build each strategy block
    for sid in (selected_strategy_ids or []):
        try:
            uid, csv_path, mode = _resolve_strategy_csv_path(sid)
        except Exception as e:
            errors.append(f"[ERROR] {e}")
            continue

        selected_uids.append(uid)
        strategy_sources[uid] = {
            "strategy_id": sid,
            "resolved_path": str(csv_path),
            "resolution_mode": mode,
        }

        # Extract factor from weights_store (same structure used in sh_layout save logic)
        # weights_store: { uid: {"factor": float}, ... }
        factor_val = 1.0
        try:
            entry = weights_store.get(uid, {}) if isinstance(weights_store, dict) else {}
            factor_val = float(entry.get("factor", 1.0))
        except Exception:
            factor_val = 1.0
        strategy_factors[uid] = round(factor_val, 6)

        # Load CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            errors.append(f"[ERROR] Failed reading CSV for uid={uid} path={csv_path}: {e}")
            continue

        # Check required columns
        missing = [c for c in REQ_COLS if c not in df.columns]
        if missing:
            errors.append(f"[SKIP] uid={uid} missing required columns: {missing}")
            continue

        # Parse open_dt
        open_dt = _parse_open_dt(df)
        df = df.copy()
        df["open_dt"] = open_dt
        
        # Parse close_dt (non-feature; used for leak-free training slicing + equity stitching)
        s_close = df["Date Closed"].astype(str).str.strip() + " " + df["Time Closed"].astype(str).str.strip()
        close_dt = pd.to_datetime(s_close, errors="coerce", dayfirst=False)
        nat_ratio = close_dt.isna().mean()
        if nat_ratio > 0.10:
            close_dt2 = pd.to_datetime(s_close, errors="coerce", dayfirst=True)
            if close_dt2.isna().mean() < nat_ratio:
                close_dt = close_dt2
        df["close_dt"] = close_dt


        # Coerce numeric fields
        df["opening_price"] = pd.to_numeric(df["Opening Price"], errors="coerce")
        df["premium"] = pd.to_numeric(df["Premium"], errors="coerce")
        df["pnl"] = pd.to_numeric(df["P/L"], errors="coerce")
        df["contracts"] = pd.to_numeric(df["No. of Contracts"], errors="coerce")
        df["margin_req"] = pd.to_numeric(df["Margin Req."], errors="coerce")
        df["opening_vix"] = pd.to_numeric(df["Opening VIX"], errors="coerce")
        df["gap"] = pd.to_numeric(df["Gap"], errors="coerce")

        # Drop rows with missing essentials
        essential = ["open_dt", "close_dt", "pnl", "contracts", "margin_req"]
        before = len(df)
        df = df.dropna(subset=essential)        
        
        dropped = before - len(df)
        if dropped > 0:
            errors.append(f"[SKIP] uid={uid} dropped {dropped} rows with NaN essentials")

        if df.empty:
            errors.append(f"[SKIP] uid={uid} has no usable rows after cleaning")
            continue

        # Derived time features
        df["open_date"] = df["open_dt"].dt.date.astype(str)
        df["close_date"] = df["close_dt"].dt.date.astype(str)
        df["dow"] = df["open_dt"].dt.dayofweek.astype(int)
        df["open_minute"] = (df["open_dt"].dt.hour * 60 + df["open_dt"].dt.minute).astype(int)

        # Strategy key
        df["strategy_uid"] = uid

        # Normalization + label
        df["risk_unit"] = df["margin_req"] * df["contracts"]
        df = df[df["risk_unit"] > 0]
        if df.empty:
            errors.append(f"[SKIP] uid={uid} all rows had non-positive risk_unit")
            continue

        df["pnl_R"] = df["pnl"] / df["risk_unit"]
        df["label"] = (df["pnl_R"] > 0).astype(int)

        # Keep only final cols
        df_final = df[FINAL_COLS].copy()
        rows_all.append(df_final)

    # Combine
    if not rows_all:
        snapshot_id = f"SNAP_{_now_stamp()}"
        # Still record snapshot attempt? In v1, no—return error to caller.
        return {
            "snapshot_id": snapshot_id,
            "dataset_path": None,
            "n_rows": 0,
            "min_open_dt": None,
            "max_open_dt": None,
            "selected_uids": selected_uids,
            "errors": errors or ["[ERROR] No valid strategy data to build dataset."],
        }

    panel = pd.concat(rows_all, ignore_index=True)

    # Sort and drop exact duplicates on (open_dt, strategy_uid)
    panel = panel.sort_values(["open_dt", "strategy_uid"]).reset_index(drop=True)
    # Optional: detect collisions (diagnostic only)
    dups = panel.duplicated(subset=["open_dt", "strategy_uid"]).sum()
    if dups > 0:
        errors.append(f"[WARN] Found {dups} rows with duplicate (open_dt, strategy_uid) keys; no rows removed.")

    # Snapshot id
    snapshot_id = f"SNAP_{_now_stamp()}"

    dataset_path: Optional[Path] = None
    dataset_sha = None

    # Persist dataset (optional)
    if save_to_disk:
        dataset_path = DATASETS_DIR / f"panel__{snapshot_id}__N{len(set(panel['strategy_uid']))}.csv"
        panel.to_csv(dataset_path, index=False, encoding="utf-8")
        dataset_sha = _file_sha256(dataset_path)

    # Snapshot entry for ml_datasets.json
    entry = {
        "snapshot_id": snapshot_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_portfolio_id": source_portfolio_id,
        "source_portfolio_name": source_portfolio_name,
        "selected_uids": sorted(list(set(selected_uids))),
        "strategy_sources": strategy_sources,      # uid -> {strategy_id, resolved_path, mode}
        "strategy_factors": strategy_factors,      # uid -> factor (from weights store)
        "rules": {
            "version": "v1",
            "feature_columns": [
                "dow", "open_minute", "opening_price", "premium",
                "margin_req", "opening_vix", "gap"
            ],
            "target_columns": ["pnl", "contracts", "risk_unit", "pnl_R", "label"],
            "label_definition": "label = 1 if pnl_R > 0 else 0",
            "risk_unit_definition": "risk_unit = margin_req * contracts",
            "leakage_policy": "Final dataset excludes all close-related columns and lifecycle extrema.",
        },
        "dataset": {
            "saved": bool(save_to_disk),
            "path": str(dataset_path) if dataset_path else None,
            "sha256": dataset_sha,
            "n_rows": int(len(panel)),
            "min_open_dt": panel["open_dt"].min().isoformat(),
            "max_open_dt": panel["open_dt"].max().isoformat(),
            "n_strategies": int(panel["strategy_uid"].nunique()),
        },
        "build_warnings": errors,
    }

    index = _load_index()
    index["datasets"].append(entry)
    _save_index(index)

    return {
        "snapshot_id": snapshot_id,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "n_rows": int(len(panel)),
        "min_open_dt": panel["open_dt"].min().isoformat(),
        "max_open_dt": panel["open_dt"].max().isoformat(),
        "selected_uids": sorted(list(set(selected_uids))),
        "errors": errors,
    }


# ------------------------------------------------------------
# Spyder / CLI test harness (safe to run standalone) for testing when run via spyder manually
# ------------------------------------------------------------
# if __name__ == "__main__":
#     # Minimal example:
#     # - Replace selected_strategy_ids with a few real CSV file paths from your app universe/active list.
#     # - Replace weights_store with something like: {"MyUID": {"factor": 1.25}, ...}
#     selected_strategy_ids = [
#         r"C:\Users\mauro\MAURO\Spyder\Portfolio26\core\data\strategies\0-DTE-Daily-AM-IC.csv",
#         r"C:\Users\mauro\MAURO\Spyder\Portfolio26\core\data\strategies\3DTE-Weekend-Condor-Early-Exit_20251216120801.csv",
#         r"C:\Users\mauro\MAURO\Spyder\Portfolio26\core\data\strategies\9-Day-IC-VIX-9D-0-90-.csv",
#         r"C:\Users\mauro\MAURO\Spyder\Portfolio26\core\data\strategies\18-45-WED-10-DTE-RIC-SMA-Crossover.csv",
#     ]

#     weights_store = {
#         # "some_strategy_uid": {"factor": 1.0},
#     }

#     result = build_ml_panel_dataset(
#         selected_strategy_ids=selected_strategy_ids,
#         weights_store=weights_store,
#         source_portfolio_id=None,
#         source_portfolio_name=None,
#         save_to_disk=True,
#     )

#     print(json.dumps(result, indent=2))

