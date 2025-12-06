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
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Registry JSON file path: same folder as this registry.py
REGISTRY_FILE = Path(__file__).with_name("registry.json")

# Internal strategy storage directory (for persistent saved strategies)
INTERNAL_STRATEGY_DIR = Path(__file__).parent / "data" / "strategies"

# Ensure the internal strategy directory exists
INTERNAL_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)


def _default_registry() -> Dict[str, Any]:
    """
    Default empty registry structure (v2).
    It is intentionally simple and extensible.
    """
    return {
        "version": 2,
        "strategies": [],  # list of strategy dicts
        "portfolios": [],  # list of portfolio dicts
    }


# ---------------------------------------------------------------------------
# UID Helpers
# ---------------------------------------------------------------------------

def derive_uid_from_filepath(file_path: str) -> str:
    """
    Derive a strategy UID from a file path.
    
    UID = filename without the .csv extension.
    Example: "/path/to/MyStrategy.csv" -> "MyStrategy"
    """
    base = os.path.basename(file_path)
    if base.lower().endswith(".csv"):
        return base[:-4]
    return base


def get_internal_strategy_path(uid: str) -> Path:
    """
    Return the internal storage path for a strategy with the given UID.
    
    The file will be stored as: INTERNAL_STRATEGY_DIR / <uid>.csv
    """
    return INTERNAL_STRATEGY_DIR / f"{uid}.csv"


def uid_exists_in_registry(uid: str, registry: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if a UID already exists in the registry.
    
    Parameters
    ----------
    uid : str
        The UID to check.
    registry : dict, optional
        Registry dict to check. If None, loads from disk.
        
    Returns
    -------
    bool
        True if the UID exists, False otherwise.
    """
    if registry is None:
        registry = load_registry()
    strategies = registry.get("strategies", [])
    for s in strategies:
        if s.get("uid") == uid:
            return True
    return False


def validate_uid_uniqueness(
    candidate_uids: List[str],
    registry: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Validate a list of candidate UIDs for uniqueness.
    
    Parameters
    ----------
    candidate_uids : list of str
        List of UIDs to validate.
    registry : dict, optional
        Registry dict to check against. If None, loads from disk.
        
    Returns
    -------
    tuple of (valid_uids, rejected_uids)
        valid_uids: List of UIDs that are unique.
        rejected_uids: List of UIDs that are duplicates.
    """
    if registry is None:
        registry = load_registry()
    
    existing_uids = set()
    for s in registry.get("strategies", []):
        uid = s.get("uid")
        if uid:
            existing_uids.add(uid)
    
    valid_uids = []
    rejected_uids = []
    seen_in_batch = set()
    
    for uid in candidate_uids:
        if uid in existing_uids or uid in seen_in_batch:
            rejected_uids.append(uid)
        else:
            valid_uids.append(uid)
            seen_in_batch.add(uid)
    
    return valid_uids, rejected_uids


def copy_strategy_to_internal(source_path: str, uid: str) -> bool:
    """
    Copy a strategy CSV file to the internal storage directory.
    
    Parameters
    ----------
    source_path : str
        Path to the source CSV file.
    uid : str
        The UID for this strategy (used as filename).
        
    Returns
    -------
    bool
        True if copy succeeded, False otherwise.
    """
    dest_path = get_internal_strategy_path(uid)
    try:
        # Ensure directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        return True
    except Exception:
        return False


def get_strategy_by_uid(uid: str, registry: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Get a strategy by its UID.
    
    Parameters
    ----------
    uid : str
        The UID to look up.
    registry : dict, optional
        Registry dict to search. If None, loads from disk.
        
    Returns
    -------
    dict or None
        The strategy dict if found, None otherwise.
    """
    if registry is None:
        registry = load_registry()
    strategies = registry.get("strategies", [])
    for s in strategies:
        if s.get("uid") == uid:
            return s
    return None


def _find_strategy_index_by_uid(strategies: List[Dict[str, Any]], uid: str) -> int:
    """
    Return index of the strategy with the given UID, or -1 if not found.
    """
    for i, s in enumerate(strategies):
        if s.get("uid") == uid:
            return i
    return -1



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

    - If the file does not exist, create it with a default registry (v2) and return that.
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
        data["version"] = 2

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
        - uid: unique identifier derived from filename without .csv extension
        - name: human-readable name
        - file_path: path to the CSV/log file for this strategy

    Optional fields:
        - id: legacy id field (kept for backward compatibility)
        - is_saved: boolean indicating if the strategy is persisted in internal storage
        - file_name: the CSV filename (<uid>.csv)

    Any non-serializable / heavy fields (e.g. 'df') are stripped
    before saving to the JSON registry.
    """
    # Work on a shallow copy and remove non-serializable fields
    strategy = dict(strategy)
    strategy.pop("df", None)  # df only lives in memory, not in JSON
    # UI state flags must NOT be persisted to disk
    strategy.pop("is_active", None)
    strategy.pop("is_selected", None)

    # Derive uid from file_path if not provided
    if "uid" not in strategy and "file_path" in strategy:
        strategy["uid"] = derive_uid_from_filepath(strategy["file_path"])
    
    # Set file_name if not provided
    if "file_name" not in strategy and "uid" in strategy:
        strategy["file_name"] = f"{strategy['uid']}.csv"
    
    # Set is_saved default to False if not provided
    if "is_saved" not in strategy:
        strategy["is_saved"] = False
    
    # Keep legacy id field for backward compatibility
    if "id" not in strategy and "file_path" in strategy:
        strategy["id"] = strategy["file_path"]

    required = ["uid", "name", "file_path"]
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

    # Use UID for lookup instead of id
    idx = _find_strategy_index_by_uid(strategies, strategy["uid"])

    if idx == -1:
        # New strategy
        strategies.append(strategy)
    else:
        # Update existing; keep previous date_added if present
        existing = strategies[idx]
        if "date_added" in existing and "date_added" in strategy:
            strategy["date_added"] = existing["date_added"]
        # Preserve is_saved if it was True (never downgrade from True to False).
        # This ensures that once a strategy has been copied to internal storage
        # (is_saved=True), it stays marked as saved even if the caller doesn't
        # know about the existing state. This prevents data inconsistency where
        # the registry says is_saved=False but the file already exists internally.
        if existing.get("is_saved", False):
            strategy["is_saved"] = True
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
    based on the list of active strategy IDs (legacy id or uid).
    """
    active_set = set(active_ids or [])

    registry = load_registry()
    strategies = registry.get("strategies", [])

    for s in strategies:
        # Check both uid and id for backward compatibility
        sid = s.get("id")
        uid = s.get("uid")
        s["phase1_active"] = (sid in active_set) or (uid in active_set)

    registry["strategies"] = strategies
    save_registry(registry)


def set_phase1_active_flags_by_uid(active_uids: List[str]) -> None:
    """
    Update the `phase1_active` flag for all strategies
    based on the list of active strategy UIDs.
    """
    active_set = set(active_uids or [])

    registry = load_registry()
    strategies = registry.get("strategies", [])

    for s in strategies:
        uid = s.get("uid")
        s["phase1_active"] = uid in active_set

    registry["strategies"] = strategies
    save_registry(registry)


def mark_strategies_as_saved(uids: List[str], source_paths: Optional[Dict[str, str]] = None) -> Tuple[List[str], List[str]]:
    """
    Mark strategies as saved and copy them to internal storage if needed.
    
    Parameters
    ----------
    uids : list of str
        List of strategy UIDs to mark as saved.
    source_paths : dict, optional
        Mapping from UID to source file path (for strategies not yet saved).
        If not provided, will try to use file_path from registry.
        
    Returns
    -------
    tuple of (success_uids, failed_uids)
        success_uids: UIDs successfully marked as saved
        failed_uids: UIDs that failed to save (e.g., copy error)
    """
    source_paths = source_paths or {}
    
    registry = load_registry()
    strategies = registry.get("strategies", [])
    
    success_uids = []
    failed_uids = []
    
    for uid in uids:
        idx = _find_strategy_index_by_uid(strategies, uid)
        if idx == -1:
            failed_uids.append(uid)
            continue
        
        strategy = strategies[idx]
        
        # Already saved - nothing to do
        if strategy.get("is_saved", False):
            success_uids.append(uid)
            continue
        
        # Get source path
        source_path = source_paths.get(uid) or strategy.get("file_path")
        if not source_path:
            failed_uids.append(uid)
            continue
        
        # Copy to internal storage
        if copy_strategy_to_internal(source_path, uid):
            strategy["is_saved"] = True
            strategies[idx] = strategy
            success_uids.append(uid)
        else:
            failed_uids.append(uid)
    
    registry["strategies"] = strategies
    save_registry(registry)
    
    return success_uids, failed_uids
    
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
        - strategy_uids: list of strategy UIDs (strings)
        
    For backward compatibility, strategy_ids is also accepted and will be
    converted to strategy_uids.
    """
    # Convert strategy_ids to strategy_uids for backward compatibility
    if "strategy_ids" in portfolio and "strategy_uids" not in portfolio:
        portfolio["strategy_uids"] = portfolio["strategy_ids"]
    
    required = ["id", "name"]
    missing = [k for k in required if k not in portfolio or not portfolio[k]]
    if missing:
        raise ValueError(f"Missing required fields in portfolio: {missing}")
    
    # Require at least one of strategy_uids or strategy_ids
    if "strategy_uids" not in portfolio and "strategy_ids" not in portfolio:
        raise ValueError("Missing required field in portfolio: strategy_uids")
    
    # Ensure strategy_uids is the canonical field
    if "strategy_uids" not in portfolio:
        portfolio["strategy_uids"] = portfolio.get("strategy_ids", [])
    
    if not isinstance(portfolio["strategy_uids"], list):
        raise ValueError("portfolio['strategy_uids'] must be a list of strategy UIDs")
    
    # Keep strategy_ids for backward compatibility
    if "strategy_ids" not in portfolio:
        portfolio["strategy_ids"] = portfolio["strategy_uids"]

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


def get_portfolios_for_uid(uid: str, registry: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Get all portfolios that contain a specific strategy UID.
    
    Parameters
    ----------
    uid : str
        The strategy UID to look up.
    registry : dict, optional
        Registry dict to search. If None, loads from disk.
        
    Returns
    -------
    list of dict
        List of portfolio dicts that contain this UID.
    """
    if registry is None:
        registry = load_registry()
    portfolios = registry.get("portfolios", [])
    matching = []
    for p in portfolios:
        strategy_uids = p.get("strategy_uids", p.get("strategy_ids", []))
        if uid in strategy_uids:
            matching.append(p)
    return matching


def get_all_saved_uids(registry: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Get all UIDs of strategies marked as saved (is_saved=True).
    
    Parameters
    ----------
    registry : dict, optional
        Registry dict to check. If None, loads from disk.
        
    Returns
    -------
    list of str
        List of UIDs for all saved strategies.
    """
    if registry is None:
        registry = load_registry()
    strategies = registry.get("strategies", [])
    saved_uids = []
    for s in strategies:
        if s.get("is_saved", False):
            uid = s.get("uid")
            if uid:
                saved_uids.append(uid)
    return saved_uids

