# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 08:57:15 2025

@author: mauro
"""

# optuna_ml2_weekly_pcr.py
# Run in Spyder. Must sit in the same folder as ml2.py (or adjust sys.path below).

from __future__ import annotations

import os
import time
import json
from datetime import datetime
from typing import Dict, Any

import io
import contextlib
import sys

import numpy as np
import pandas as pd
import optuna

# ------------------------------------------------------------
# IMPORT YOUR EXISTING ENGINE (NO MODIFICATIONS TO ml2.py)
# ------------------------------------------------------------
import ml2


# ------------------------------------------------------------
# HARD-CODED DATASET PATH (YOU PROVIDED THIS)
# ------------------------------------------------------------
DATASET_CSV = r"C:\Users\mauro\MAURO\Spyder\Portfolio26\ml\datasets\panel__SNAP_20251218_194959__N9.csv"


# ------------------------------------------------------------
# RUN PARAMS (YOU SAID YOU WILL EDIT THESE AS NEEDED)
# ------------------------------------------------------------
RUN_PARAMS = dict(
    dataset_csv_path=DATASET_CSV,
    start_date      =   None,         # e.g. "2024-01-01" or None
    end_date        =   None,          # e.g. "2025-12-01" or None
    is_months       =   2,
    oos_weeks       =   1,
    step_weeks      =   1,
    anchored_type   =   "U",      # "U" or "A"
    top_k_per_day   =   3,
    verbose_cycles  =   False,   # keep False for Optuna speed; set True for debugging
    initial_equity  =   100000.0,
    debug_cycle_to_print=None,
    debug_max_rows=40,
)


# ------------------------------------------------------------
# OPTUNA SETTINGS (YOU SAID YOU WILL EDIT THESE)
# ------------------------------------------------------------
N_TRIALS = 10

# Use RandomSampler so the "seed" parameter behaves as a truly random draw
# (and Optuna does not try to "optimize" the seed). You can switch to TPE later.
SAMPLER = optuna.samplers.RandomSampler()

# If you want to keep a study DB on disk (resume later), set STORAGE path.
# Leave as None for in-memory.
STORAGE = None  # e.g. "sqlite:///optuna_ml2_weekly_pcr.db"
STUDY_NAME = "ml2_weekly_pcr"


# ------------------------------------------------------------
# HYPERPARAMETER SPEC (range + step in the same place)
# - Use step for linear grids
# - Use log=True for log-scale (no step allowed by Optuna in log mode)
# - Use choices for categorical grids (discrete list)
# ------------------------------------------------------------
HP_SPEC = {
    # ---- Core boosting
    "n_estimators":        {"low": 45,  "high": 400,  "step": 5},     # int grid
    "learning_rate":       {"low": 0.01, "high": 0.15, "step": 0.01},    # log-scale (no step)

    # ---- Tree shape
    #"num_leaves":          {"choices": [7, 15, 31, 63]},                # categorical grid
    "num_leaves":          {"low": 15,    "high": 50,   "step": 1},               # categorical grid
    "min_child_samples":   {"low": 5,    "high": 80,   "step": 5},      # int grid
    "max_depth":           {"choices": [-1, 3, 5, 7, 9, 12]},           # categorical grid

    # ---- Sampling
    "subsample":           {"low": 0.20, "high": 1.00, "step": 0.02},   # float grid
    "colsample_bytree":    {"low": 0.40, "high": 1.00, "step": 0.02},   # float grid

    # ---- Regularization
    "reg_alpha":           {"low": 0.0,  "high": 10.0, "step": 0.5},    # float grid
    "reg_lambda":          {"low": 0.0,  "high": 10.0, "step": 0.5},    # float grid

    # ---- Split gain threshold (canonical LightGBM param name)
    "min_gain_to_split":   {"low": 0.0,  "high": 1.0,  "step": 0.05},   # float grid
}

def suggest_from_spec(trial: optuna.Trial, name: str):
    spec = HP_SPEC[name]

    # Categorical
    if "choices" in spec:
        return trial.suggest_categorical(name, spec["choices"])

    low = spec["low"]
    high = spec["high"]
    log = bool(spec.get("log", False))
    step = spec.get("step", None)

    # Int
    if isinstance(low, int) and isinstance(high, int):
        if step is None:
            return trial.suggest_int(name, low, high, log=log)
        return trial.suggest_int(name, low, high, step=int(step), log=log)

    # Float
    if step is None:
        return trial.suggest_float(name, float(low), float(high), log=log)

    # Optuna does not allow step + log=True for floats
    if log:
        raise ValueError(f"{name}: step is not allowed with log=True for floats in Optuna.")
    return trial.suggest_float(name, float(low), float(high), step=float(step), log=False)


# ------------------------------------------------------------
# OUTPUT LOG
# ------------------------------------------------------------
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
#OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = r"C:\Users\mauro\MAURO\Options Trading\TradeBusters\Portfolio26\Phase3"
CSV_LOG_PATH = os.path.join(OUT_DIR, f"optuna_ml2_weekly_pcr__{RUN_TAG}.csv")


def _safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _build_lgbm_params(trial: optuna.Trial, seed: int) -> Dict[str, Any]:
    p = {}
    p["boosting_type"] = "gbdt"
    p["class_weight"] = "balanced"
    p["n_jobs"] = -1
    p["deterministic"] = True

    # ---- suggested params from HP_SPEC
    p["n_estimators"] = suggest_from_spec(trial, "n_estimators")
    p["learning_rate"] = suggest_from_spec(trial, "learning_rate")

    p["num_leaves"] = suggest_from_spec(trial, "num_leaves")
    p["min_child_samples"] = suggest_from_spec(trial, "min_child_samples")
    p["max_depth"] = suggest_from_spec(trial, "max_depth")

    p["subsample"] = suggest_from_spec(trial, "subsample")
    p["colsample_bytree"] = suggest_from_spec(trial, "colsample_bytree")

    p["reg_alpha"] = suggest_from_spec(trial, "reg_alpha")
    p["reg_lambda"] = suggest_from_spec(trial, "reg_lambda")

    p["min_gain_to_split"] = suggest_from_spec(trial, "min_gain_to_split")

    # ---- Single global seed (features + model)
    p["random_state"] = int(seed)

    # ---- Silence LightGBM console spam
    p["verbosity"] = -1
    p["force_col_wise"] = True

    return p


@contextlib.contextmanager
def suppress_output(enabled: bool = True):
    """
    Suppress ALL stdout/stderr (print spam, cycle logs, library noise).
    Does NOT modify ml2.py.
    """
    if not enabled:
        yield
        return

    devnull_out = io.StringIO()
    devnull_err = io.StringIO()
    with contextlib.redirect_stdout(devnull_out), contextlib.redirect_stderr(devnull_err):
        yield


def _flatten_metrics(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, (int, float, np.integer, np.floating)) or v is None:
            out[key] = _safe_float(v)
        else:
            # keep non-numerics as JSON strings (rare in your metrics dict, but safe)
            try:
                out[key] = json.dumps(v)
            except Exception:
                out[key] = str(v)
    return out


def _append_csv_row(row: Dict[str, Any], csv_path: str) -> None:
    df_row = pd.DataFrame([row])
    if not os.path.exists(csv_path):
        df_row.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, mode="a", header=False, index=False)


def objective(trial: optuna.Trial) -> float:
    # Seed must be random per trial, but tracked so best trial is reproducible.
    seed = trial.suggest_int("seed", 1, 2_000_000_000)

    # Apply the same seed to BOTH feature generation and LGBM.
    ml2.RANDOM_SEED = int(seed)

    # Build RunParamsWeekly using your dataclass
    params = ml2.RunParamsWeekly(**RUN_PARAMS)
    params.lgbm_params = _build_lgbm_params(trial, seed=seed)

    t0 = time.time()
    try:
        with suppress_output(enabled=True):
            out = ml2.run_fwa_weekly(params)
    except Exception as e:
        # Hard fail this trial but keep study running.
        # You will see the exception text in Optuna log.
        raise optuna.TrialPruned(f"Trial failed: {repr(e)}")

    elapsed = time.time() - t0

    ml_metrics = out.get("ml_metrics", {}) or {}
    base_metrics = out.get("baseline_metrics", {}) or {}

    # Your requested objective: maximize ML PCR
    pcr = ml_metrics.get("pcr", np.nan)

    # If PCR is NaN (e.g., no trades), treat as a very bad score.
    score = _safe_float(pcr)
    if not np.isfinite(score):
        score = -1e9

    # Log row to CSV
    row = {
        "trial": int(trial.number),
        "seed": int(seed),
        "elapsed_sec": float(elapsed),
    }
    row.update({k: v for k, v in trial.params.items()})  # includes hyperparams + seed

    # Attach both stitched metric dicts from ml2.py
    row.update(_flatten_metrics("ml_", ml_metrics))
    row.update(_flatten_metrics("base_", base_metrics))

    _append_csv_row(row, CSV_LOG_PATH)

    # Console one-liner (keep tight)
    print(
        f"[trial={trial.number:04d}] score(PCR)={score:.6f} "
        f"ml_pnl={_safe_float(ml_metrics.get('total_pnl', np.nan)):.2f} "
        f"ml_dd$={_safe_float(ml_metrics.get('max_dd_$', np.nan)):.2f} "
        f"ml_sh={_safe_float(ml_metrics.get('sharpe_daily', np.nan)):.3f} "
        f"seed={seed} elapsed={elapsed:.2f}s"
    )

    return score


def main() -> None:
    print("=== Optuna ML2 Weekly CPO (maximize ML PCR) ===")
    print(f"Dataset: {DATASET_CSV}")
    print(f"CSV log: {CSV_LOG_PATH}")
    print(f"N_TRIALS: {N_TRIALS}")
    print(f"Study: {STUDY_NAME}")
    print("")

    if STORAGE:
        study = optuna.create_study(
            direction="maximize",
            sampler=SAMPLER,
            study_name=STUDY_NAME,
            storage=STORAGE,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=SAMPLER,
            study_name=STUDY_NAME,
        )

    study.optimize(objective, n_trials=int(N_TRIALS), gc_after_trial=True)

    bt = study.best_trial
    print("\n=== BEST TRIAL ===")
    print(f"Best value (PCR): {bt.value}")
    print("Best params:")
    for k, v in bt.params.items():
        print(f"  {k}: {v}")

    print(f"\nCSV log written: {CSV_LOG_PATH}")


if __name__ == "__main__":
    main()
