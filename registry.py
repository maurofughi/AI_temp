# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 08:20:53 2025

@author: mauro
"""

# -*- coding: utf-8 -*-
"""
Central registry for strategies and portfolios used in the app.

- Keeps a JSON file `registry.json` next to this module.
- API is backward-compatible with the previous version:
  - list_strategies(), get_strategy(), add_or_update_strategy(),
    remove_strategy(), set_phase1_active_flags()
  - list_portfolios(), get_portfolio(), add_or_update_portfolio()
- New in schema version 2:
  - Strategies get a persistent numeric UID (string, e.g. "S000001") independent
    of file path.
  - Portfolios can store per-strategy `weights` and `size_factors` keyed by the
    same `strategy_ids` list currently used (file paths).
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# Registry JSON file path: same folder as this registry.py
REGISTRY_FILE = Path(__file__).with_name("registry.json")


# ---------------------------------------------------------------------------
# Default / schema helpers
# ---------------------------------------------------------------------------

def _default_registry() -> Dict[str, Any]:
    """
    Default empty registry structure.

    Schema v2:
    - version: 2
    - next_strategy_uid: numeric counter for strategy UIDs
    - next_portfolio_uid: numeric counter for portfolio UIDs (reserved for later)
    - strategies: list of strategy dicts
    - portfolios: list of portfolio dicts
    """
    return {
        "version": 2,
        "next_strategy_uid": 1,
        "next_portfolio_uid": 1,
        "strategies": [],   # list of strategy dicts
        "portfolios": [],   # list of portfolio dicts
    }


def _ensure_registry_file_exists() -> None:
    """Create an empty registry file if it does not exist yet."""
    if not REGISTRY_FILE.exists():
        registry = _default_registry()
        save_registry(registry)


def _ensure_schema_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure that the loaded registry conforms to schema version 2.

    - If 'version' is missing or < 2, we upgrade in-place:
      - set version = 2
      - ensure next_strategy_uid / next_portfolio_uid exist
      - ensure each strategy has a 'uid'
      - (optional) each portfolio gets a 'uid' counter for future use
    """
    if not isinstance(data, dict):
        return _default_registry()

    version = data.get("version", 1)
    if version < 2:
        data["version"] = 2

    # Ensure counters
    if not isinstance(data.get("next_strategy_uid"), int):
        data["next_strategy_uid"] = 1
    if not isinstance(data.get("next_portfolio_uid"), int):
        data["next_portfolio_uid"] = 1

    # Ensure list containers
    if not isinstance(data.get("strategies"), list):
        data["strategies"] = []
    if not isinstance(data.get("portfolios"), list):
        data["portfolios"] = []

    # Assign UIDs to strategies that do not have one yet
    next_sid = data["next_strategy_uid"]
    for s in data["strategies"]:
        if "uid" not in s or not s["uid"]:
            s["uid"] = f"S{next_sid:06d}"
            next_sid += 1
    data["next_strategy_uid"] = next_sid

    # Reserve portfolio UIDs for the future (not used anywhere yet)
    next_pid = data["next_portfolio_uid"]
    for p in data["portfolios"]:
        if "uid" not in p or not p["uid"]:
            p["uid"] = f"P{next_pid:06d}"
            next_pid += 1
    data["next_portfolio_uid"] = next_pid

    return data


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_registry() -> Dict[str, Any]:
    """
    Load the registry from disk.

    - If the file does not exist, create it with a default registry and return that.
    - If the file is temporarily unreadable/corrupted (e.g. partial write),
      we do NOT overwrite the file; we just fall back to a default in memory.
    - Always normalises to schema version 2.
    """
    _ensure_registry_file_exists()
    try:
        with REGISTRY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Do NOT overwrite the file here; just return a default structure
        data = _default_registry()

    # Normalise to v2
    data = _ensure_schema_v2(data)
    return data


def save_registry(registry: Dict[str, Any]) -> None:
    """
    Save the registry to disk.

    Preferred path: atomic write via temp file + replace.
    On Windows, if os.replace / Path.replace raises PermissionError (file locked
    by AV/editor/other process), fall back to a direct write.
    """
    # Always ensure schema is coherent before saving
    if not isinstance(registry, dict):
        registry = _default_registry()
    registry = _ensure_schema_v2(registry)

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


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def list_strategies() -> List[Dict[str, Any]]:
    """Return the list of strategies from the registry."""
    registry = load_registry()
    return registry.get("strategies", [])


def _find_strategy_index(strategies: List[Dict[str, Any]], strategy_id: str) -> int:
    """Return index of the strategy with the given id (path), or -1 if not found."""
    for i, s in enumerate(strategies):
        if s.get("id") == strategy_id:
            return i
    return -1


def get_strategy(strategy_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a single strategy by `id`.

    NOTE: `strategy_id` is still the DASH-facing id (currently the file path).
    """
    strategies = list_strategies()
    idx = _find_strategy_index(strategies, strategy_id)
    if idx == -1:
        return None
    return strategies[idx]


def _next_strategy_uid(registry: Dict[str, Any]) -> str:
    """Allocate the next strategy UID and increment the counter."""
    n = registry.get("next_strategy_uid", 1)
    uid = f"S{n:06d}"
    registry["next_strategy_uid"] = n + 1
    return uid


def add_or_update_strategy(strategy: Dict[str, Any]) -> None:
    """
    Add a new strategy or update an existing one.

    Required fields in `strategy` (call-site remains unchanged):
      - id: unique identifier (string)  [currently the file path]
      - name: human-readable name
      - file_path: path to the CSV/log file for this strategy

    Any non-serializable / heavy fields (e.g. 'df') are stripped before
    saving to the JSON registry.

    New in schema v2:
      - Each strategy gets a 'uid' (S000001-style) that is persistent
        and independent of the file path.
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
    now_iso = datetime.now().isoformat(timespec="seconds")

    # Ensure we keep some metadata fields if present at call-site
    # (folder, source, n_rows etc. are optional).
    incoming_id = strategy["id"]            # DASH-facing id (path for now)
    file_path = strategy["file_path"]
    name = strategy["name"]
    folder = strategy.get("folder")
    source = strategy.get("source")
    n_rows = strategy.get("n_rows")
    phase1_active = strategy.get("phase1_active", False)

    # See if this strategy already exists by its DASH id (path)
    idx = _find_strategy_index(strategies, incoming_id)

    if idx == -1:
        # New strategy
        uid = _next_strategy_uid(registry)
        new_s = {
            "uid": uid,
            "id": incoming_id,         # keep for compatibility with existing UI
            "name": name,
            "file_path": file_path,
            "folder": folder,
            "source": source,
            "n_rows": n_rows,
            "phase1_active": phase1_active,
            "date_added": now_iso,
            "date_updated": now_iso,
        }
        strategies.append(new_s)
    else:
        # Update existing; keep previous date_added and uid if present
        existing = strategies[idx]
        uid = existing.get("uid") or _next_strategy_uid(registry)

        updated = {
            "uid": uid,
            "id": incoming_id,
            "name": name,
            "file_path": file_path,
            "folder": folder if folder is not None else existing.get("folder"),
            "source": source if source is not None else existing.get("source"),
            "n_rows": n_rows if n_rows is not None else existing.get("n_rows"),
            "phase1_active": existing.get("phase1_active", False),
            "date_added": existing.get("date_added", now_iso),
            "date_updated": now_iso,
        }
        strategies[idx] = updated

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
    Update the `phase1_active` flag for all strategies based on the list
    of active strategy IDs.

    NOTE: `active_ids` are still the DASH-facing ids (paths). We deliberately
    keep this behaviour so Phase 1 remains unchanged.
    """
    active_set = set(active_ids or [])
    registry = load_registry()
    strategies = registry.get("strategies", [])

    for s in strategies:
        sid = s.get("id")
        s["phase1_active"] = sid in active_set

    registry["strategies"] = strategies
    save_registry(registry)


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------

def list_portfolios() -> List[Dict[str, Any]]:
    """Return the list of portfolios from the registry."""
    registry = load_registry()
    portfolios = registry.get("portfolios", [])
    if not isinstance(portfolios, list):
        portfolios = []
    return portfolios


def _find_portfolio_index(portfolios: List[Dict[str, Any]], portfolio_id: str) -> int:
    """Return index of the portfolio with the given id, or -1 if not found."""
    for i, p in enumerate(portfolios):
        if p.get("id") == portfolio_id:
            return i
    return -1


def get_portfolio(portfolio_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a single portfolio by id.

    Returns None if not found.
    """
    portfolios = list_portfolios()
    idx = _find_portfolio_index(portfolios, portfolio_id)
    if idx == -1:
        return None
    return portfolios[idx]


def add_or_update_portfolio(portfolio: Dict[str, Any]) -> None:
    """
    Add a new portfolio or update an existing one.

    Required fields (same as before):
      - id: unique portfolio identifier (string)
      - name: human-readable portfolio name
      - strategy_ids: list of strategy ids (strings) – these are still the
        DASH-facing strategy ids (paths).

    New in schema v2:
      - Per-strategy `weights` and `size_factors`, stored as dicts keyed by
        the same `strategy_ids` (paths). If not provided, they are initialised
        to 1.0 for all strategies.
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

    # Ensure we have weights and size_factors keyed by strategy_ids (paths)
    strategy_ids = portfolio["strategy_ids"]

    weights = portfolio.get("weights")
    if not isinstance(weights, dict):
        # Initialise all weights to 1.0
        weights = {sid: 1.0 for sid in strategy_ids}
    else:
        # Ensure every strategy_id has some weight; default 1.0 if missing
        for sid in strategy_ids:
            weights.setdefault(sid, 1.0)

    size_factors = portfolio.get("size_factors")
    if not isinstance(size_factors, dict):
        size_factors = {sid: 1.0 for sid in strategy_ids}
    else:
        for sid in strategy_ids:
            size_factors.setdefault(sid, 1.0)

    portfolio["weights"] = weights
    portfolio["size_factors"] = size_factors

    # Ensure portfolio UID (reserved, not used by UI yet)
    if "uid" not in portfolio or not portfolio["uid"]:
        # Allocate a UID if needed
        pid_counter = registry.get("next_portfolio_uid", 1)
        portfolio["uid"] = f"P{pid_counter:06d}"
        registry["next_portfolio_uid"] = pid_counter + 1

    # Timestamps
    if "created_at" not in portfolio:
        portfolio["created_at"] = now_iso
    portfolio["updated_at"] = now_iso

    idx = _find_portfolio_index(portfolios, portfolio["id"])
    if idx == -1:
        portfolios.append(portfolio)
    else:
        existing = portfolios[idx]
        # Preserve original created_at if present
        if "created_at" in existing and "created_at" in portfolio:
            portfolio["created_at"] = existing["created_at"]
        # Preserve existing uid if present
        if "uid" in existing and existing["uid"]:
            portfolio["uid"] = existing["uid"]
        portfolios[idx] = portfolio

    registry["portfolios"] = portfolios
    save_registry(registry)
