# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 08:20:53 2025

@author: mauro
"""

# -*- coding: utf-8 -*-
"""
Central registry for strategies and portfolios in the Portfolio26 app.

This module is the *only* place that reads/writes `registry.json`.

Key design points (new schema, Dec 2025):
- Strategies have synthetic UIDs ("S000001", ...), *never* external paths.
- Portfolios reference strategies only by UID.
- External folder paths are kept only as optional metadata (source_path, source_folder).
- The app's data directory (internal_path) is the only persistent source of CSVs.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths & globals
# ---------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(THIS_DIR, "registry.json")

# Global in-memory cache
_strategy_registry: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Helpers: loading / saving
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _empty_registry() -> Dict[str, Any]:
    return {
        "next_strategy_uid": 1,
        "next_portfolio_uid": 1,
        "strategies": [],
        "portfolios": [],
    }


def _load_registry_from_disk() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY_PATH):
        return _empty_registry()
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Basic sanity checks; if missing keys, re-initialise.
        if not isinstance(data, dict):
            return _empty_registry()
        if "strategies" not in data or "portfolios" not in data:
            return _empty_registry()
        if "next_strategy_uid" not in data or "next_portfolio_uid" not in data:
            data.setdefault("next_strategy_uid", 1)
            data.setdefault("next_portfolio_uid", 1)
        return data
    except Exception:
        # On any parse error, start fresh (better than crashing the app).
        return _empty_registry()


def _ensure_loaded() -> None:
    global _strategy_registry
    if not _strategy_registry:
        _strategy_registry = _load_registry_from_disk()


def save_registry(registry: Optional[Dict[str, Any]] = None) -> None:
    """
    Persist the current registry to disk.

    If `registry` is provided, it becomes the global object.
    """
    global _strategy_registry
    if registry is not None:
        _strategy_registry = registry

    if not _strategy_registry:
        _strategy_registry = _empty_registry()

    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(_strategy_registry, f, indent=2, sort_keys=False)


def get_registry() -> Dict[str, Any]:
    """
    Get the live registry dict (loaded once and cached).
    """
    _ensure_loaded()
    return _strategy_registry


# ---------------------------------------------------------------------------
# UID generators
# ---------------------------------------------------------------------------

def _format_uid(prefix: str, num: int) -> str:
    return f"{prefix}{num:06d}"


def _next_strategy_uid(registry: Dict[str, Any]) -> str:
    n = registry.get("next_strategy_uid", 1)
    uid = _format_uid("S", n)
    registry["next_strategy_uid"] = n + 1
    return uid


def _next_portfolio_uid(registry: Dict[str, Any]) -> str:
    n = registry.get("next_portfolio_uid", 1)
    uid = _format_uid("P", n)
    registry["next_portfolio_uid"] = n + 1
    return uid


# ---------------------------------------------------------------------------
# Strategy operations
# ---------------------------------------------------------------------------

def list_strategies() -> List[Dict[str, Any]]:
    reg = get_registry()
    return reg.get("strategies", [])


def _find_strategy_index(strategies: List[Dict[str, Any]], uid: str) -> int:
    for i, s in enumerate(strategies):
        if s.get("uid") == uid:
            return i
    return -1


def get_strategy(uid: str) -> Optional[Dict[str, Any]]:
    """
    Get a single strategy by UID. Returns None if not found.
    """
    reg = get_registry()
    strategies = reg.get("strategies", [])
    idx = _find_strategy_index(strategies, uid)
    if idx == -1:
        return None
    return strategies[idx]


def add_or_update_strategy(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert or update a strategy in the registry.

    Expected fields in `strategy`:
        - uid (optional on insert; ignored on update if present)
        - name
        - internal_path
        - source_path (optional)
        - source_folder (optional)
        - n_rows (optional)
        - phase1_active (optional)
    """
    reg = get_registry()
    strategies = reg.get("strategies", [])

    uid = strategy.get("uid")
    now = _now_iso()

    if uid:
        # Update existing if found
        idx = _find_strategy_index(strategies, uid)
        if idx != -1:
            existing = strategies[idx]
            created = existing.get("created_at")
            merged = dict(existing)
            merged.update(strategy)
            merged["uid"] = uid
            merged["id"] = uid  # keep legacy "id" field equal to uid
            merged["updated_at"] = now
            if created:
                merged["created_at"] = created
            strategies[idx] = merged
        else:
            # uid provided but not found; treat as new
            strategy_uid = uid
            strategy["uid"] = strategy_uid
            strategy["id"] = strategy_uid
            strategy.setdefault("created_at", now)
            strategy["updated_at"] = now
            strategies.append(strategy)
    else:
        # New strategy, assign UID
        strategy_uid = _next_strategy_uid(reg)
        strategy["uid"] = strategy_uid
        strategy["id"] = strategy_uid
        strategy.setdefault("created_at", now)
        strategy["updated_at"] = now
        strategies.append(strategy)

    reg["strategies"] = strategies
    save_registry(reg)
    return get_strategy(strategy.get("uid", strategy_uid))


def set_phase1_active_flags(active_uids: List[str]) -> None:
    """
    Mark which strategies are currently active in Phase 1.

    Any strategy whose uid is in `active_uids` will have phase1_active=True;
    all others will have phase1_active=False.
    """
    reg = get_registry()
    strategies = reg.get("strategies", [])
    active_set = set(active_uids or [])
    for s in strategies:
        s["phase1_active"] = s.get("uid") in active_set
    reg["strategies"] = strategies
    save_registry(reg)


def get_phase1_active_uids() -> List[str]:
    """
    Return list of strategy UIDs currently flagged as active in Phase 1.
    """
    strategies = list_strategies()
    return [s["uid"] for s in strategies if s.get("phase1_active")]


# ---------------------------------------------------------------------------
# Portfolio operations
# ---------------------------------------------------------------------------

def list_portfolios() -> List[Dict[str, Any]]:
    reg = get_registry()
    return reg.get("portfolios", [])


def _find_portfolio_index(portfolios: List[Dict[str, Any]], uid: str) -> int:
    for i, p in enumerate(portfolios):
        if p.get("uid") == uid:
            return i
    return -1


def get_portfolio(uid: str) -> Optional[Dict[str, Any]]:
    """
    Get a single portfolio by UID. Returns None if not found.
    """
    reg = get_registry()
    portfolios = reg.get("portfolios", [])
    idx = _find_portfolio_index(portfolios, uid)
    if idx == -1:
        return None
    return portfolios[idx]


def add_or_update_portfolio(portfolio: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert or update a portfolio in the registry.

    Expected fields:
        - uid (optional for new portfolios)
        - name (string)
        - strategy_uids (list of strategy UIDs)
        - weights (dict uid -> float)
        - size_factors (dict uid -> float)
        - phase1_done (bool, optional)
        - optimizer_settings (dict, optional)
    """
    reg = get_registry()
    portfolios = reg.get("portfolios", [])

    uid = portfolio.get("uid")
    now = _now_iso()

    if uid:
        # Update existing if found
        idx = _find_portfolio_index(portfolios, uid)
        if idx != -1:
            existing = portfolios[idx]
            created = existing.get("created_at")
            merged = dict(existing)
            merged.update(portfolio)
            merged["uid"] = uid
            merged["id"] = uid  # keep legacy "id" field equal to uid
            merged["updated_at"] = now
            if created:
                merged["created_at"] = created
            portfolios[idx] = merged
        else:
            # uid provided but not found; treat as new
            portfolio_uid = uid
            portfolio["uid"] = portfolio_uid
            portfolio["id"] = portfolio_uid
            portfolio.setdefault("created_at", now)
            portfolio["updated_at"] = now
            portfolios.append(portfolio)
    else:
        # New portfolio, assign UID
        portfolio_uid = _next_portfolio_uid(reg)
        portfolio["uid"] = portfolio_uid
        portfolio["id"] = portfolio_uid
        portfolio.setdefault("created_at", now)
        portfolio["updated_at"] = now
        portfolios.append(portfolio)

    reg["portfolios"] = portfolios
    save_registry(reg)
    return get_portfolio(portfolio.get("uid", portfolio_uid))


# ---------------------------------------------------------------------------
# Convenience helpers for UI
# ---------------------------------------------------------------------------

def list_portfolio_options() -> List[Dict[str, Any]]:
    """
    Helper to build dropdown options for portfolios.

    Returns a list of {label, value} dicts suitable for a dcc.Dropdown.
    """
    portfolios = list_portfolios()
    opts: List[Dict[str, Any]] = []
    for p in portfolios:
        label = p.get("name") or p.get("uid")
        value = p.get("uid")
        opts.append({"label": label, "value": value})
    return opts
