# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 08:20:53 2025

@author: mauro
"""

"""
Central registry for strategies loaded in the app.

This module exposes a single shared dictionary:
    strategy_registry

All pages should import and mutate this object, not create their own copies.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# Registry JSON file path: same folder as this registry.py
REGISTRY_FILE = Path(__file__).with_name("registry.json")


def _default_registry() -> Dict[str, Any]:
    """
    Default empty registry structure.
    It is intentionally simple and extensible.
    """
    return {
        "version": 1,
        "strategies": [],  # list of strategy dicts
        "portfolios": [],  # list of portfolio dicts
    }



def _ensure_registry_file_exists() -> None:
    """
    Create an empty registry file if it does not exist yet.
    """
    if not REGISTRY_FILE.exists():
        registry = _default_registry()
        save_registry(registry)


def load_registry() -> Dict[str, Any]:
    """
    Load the registry from disk.

    - If the file does not exist, create it with a default registry and return that.
    - If the file is temporarily unreadable/corrupted (e.g. partial write),
      we do NOT overwrite the file; we just fall back to a default in memory.
    """
    _ensure_registry_file_exists()

    try:
        with REGISTRY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Do NOT overwrite the file here; just return a default structure
        data = _default_registry()

    # Basic sanity: ensure dict
    if not isinstance(data, dict):
        data = _default_registry()

    # Ensure expected keys/types
    if not isinstance(data.get("strategies"), list):
        data["strategies"] = []

    if not isinstance(data.get("portfolios"), list):
        data["portfolios"] = []

    if "version" not in data:
        data["version"] = 1

    return data




def save_registry(registry: Dict[str, Any]) -> None:
    """
    Save the registry to disk.

    Preferred path: atomic write via temp file + replace.
    On Windows, if os.replace / Path.replace raises PermissionError
    (file locked by AV/editor/other process), fall back to a direct write.
    """
    tmp_path = REGISTRY_FILE.with_suffix(REGISTRY_FILE.suffix + ".tmp")

    try:
        # 1) Write to temp file
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())

        # 2) Try atomic replace
        try:
            tmp_path.replace(REGISTRY_FILE)
        except PermissionError:
            # Fallback: direct write to target file if replace is blocked
            with REGISTRY_FILE.open("w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2, sort_keys=True)

    finally:
        # 3) Best-effort cleanup of temp file if it still exists
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            # Ignore cleanup errors
            pass



def list_strategies() -> List[Dict[str, Any]]:
    """
    Convenience wrapper: return the list of strategies from the registry.
    """
    registry = load_registry()
    return registry.get("strategies", [])


def _find_strategy_index(strategies: List[Dict[str, Any]], strategy_id: str) -> int:
    """
    Return index of the strategy with the given id, or -1 if not found.
    """
    for i, s in enumerate(strategies):
        if s.get("id") == strategy_id:
            return i
    return -1


def get_strategy(strategy_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a single strategy by id. Returns None if not found.
    """
    strategies = list_strategies()
    idx = _find_strategy_index(strategies, strategy_id)
    if idx == -1:
        return None
    return strategies[idx]


def add_or_update_strategy(strategy: Dict[str, Any]) -> None:
    """
    Add a new strategy or update an existing one.

    Required fields in 'strategy':
        - id: unique identifier (string)
        - name: human-readable name
        - file_path: path to the CSV/log file for this strategy

    Any non-serializable / heavy fields (e.g. 'df') are stripped
    before saving to the JSON registry.
    """
    # Work on a shallow copy and remove non-serializable fields
    strategy = dict(strategy)
    strategy.pop("df", None)  # df only lives in memory, not in JSON

    required = ["id", "name", "file_path"]
    missing = [k for k in required if k not in strategy or not strategy[k]]
    if missing:
        raise ValueError(f"Missing required fields in strategy: {missing}")

    registry = load_registry()
    strategies = registry.get("strategies", [])


    # set metadata if not already present
    now_iso = datetime.now().isoformat(timespec="seconds")
    if "date_added" not in strategy:
        strategy["date_added"] = now_iso
    # date_updated is always refreshed
    strategy["date_updated"] = now_iso

    idx = _find_strategy_index(strategies, strategy["id"])

    if idx == -1:
        # New strategy
        strategies.append(strategy)
    else:
        # Update existing; keep previous date_added if present
        existing = strategies[idx]
        if "date_added" in existing and "date_added" in strategy:
            strategy["date_added"] = existing["date_added"]
        strategies[idx] = strategy

    registry["strategies"] = strategies
    save_registry(registry)


def remove_strategy(strategy_id: str) -> bool:
    """
    Remove a strategy by id.
    Returns True if something was removed, False if the id was not found.
    """
    registry = load_registry()
    strategies = registry.get("strategies", [])
    idx = _find_strategy_index(strategies, strategy_id)
    if idx == -1:
        return False

    del strategies[idx]
    registry["strategies"] = strategies
    save_registry(registry)
    return True


def clear_registry(confirm: bool = False) -> None:
    """
    Wipe the registry completely (for testing / debugging).
    You must call with confirm=True to avoid accidental use.
    """
    if not confirm:
        raise RuntimeError("Refusing to clear registry without confirm=True.")
    save_registry(_default_registry())

def set_phase1_active_flags(active_ids: List[str]) -> None:
    """
    Update the `phase1_active` flag for all strategies
    based on the list of active strategy IDs.
    """
    active_set = set(active_ids or [])

    registry = load_registry()
    strategies = registry.get("strategies", [])

    for s in strategies:
        sid = s.get("id")
        s["phase1_active"] = sid in active_set

    registry["strategies"] = strategies
    save_registry(registry)
    
def list_portfolios() -> List[Dict[str, Any]]:
    """
    Return the list of portfolios from the registry.
    """
    registry = load_registry()
    portfolios = registry.get("portfolios", [])
    if not isinstance(portfolios, list):
        portfolios = []
    return portfolios


def _find_portfolio_index(portfolios: List[Dict[str, Any]], portfolio_id: str) -> int:
    """
    Return index of the portfolio with the given id, or -1 if not found.
    """
    for i, p in enumerate(portfolios):
        if p.get("id") == portfolio_id:
            return i
    return -1


def get_portfolio(portfolio_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a single portfolio by id. Returns None if not found.
    """
    portfolios = list_portfolios()
    idx = _find_portfolio_index(portfolios, portfolio_id)
    if idx == -1:
        return None
    return portfolios[idx]


def add_or_update_portfolio(portfolio: Dict[str, Any]) -> None:
    """
    Add a new portfolio or update an existing one.

    Required fields:
        - id: unique portfolio identifier (string)
        - name: human-readable portfolio name
        - strategy_ids: list of strategy ids (strings)
    """
    required = ["id", "name", "strategy_ids"]
    missing = [k for k in required if k not in portfolio or not portfolio[k]]
    if missing:
        raise ValueError(f"Missing required fields in portfolio: {missing}")

    if not isinstance(portfolio["strategy_ids"], list):
        raise ValueError("portfolio['strategy_ids'] must be a list of strategy ids")

    registry = load_registry()
    portfolios = registry.get("portfolios", [])
    if not isinstance(portfolios, list):
        portfolios = []

    now_iso = datetime.now().isoformat(timespec="seconds")
    if "created_at" not in portfolio:
        portfolio["created_at"] = now_iso
    portfolio["updated_at"] = now_iso

    idx = _find_portfolio_index(portfolios, portfolio["id"])

    if idx == -1:
        portfolios.append(portfolio)
    else:
        existing = portfolios[idx]
        if "created_at" in existing and "created_at" in portfolio:
            portfolio["created_at"] = existing["created_at"]
        portfolios[idx] = portfolio

    registry["portfolios"] = portfolios
    save_registry(registry)

