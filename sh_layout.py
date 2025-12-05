# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 06:27:27 2025

@author: mauro
"""

# core/sh_layout.py

import os
import io
import base64
import uuid
import pandas as pd
from datetime import datetime
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, ctx, ALL, no_update

from core.registry import (
    load_registry,
    add_or_update_strategy,
    list_strategies,
    set_phase1_active_flags,
    list_portfolios,
    get_portfolio,
    add_or_update_portfolio,
    derive_uid_from_filepath,
    uid_exists_in_registry,
    validate_uid_uniqueness,
    mark_strategies_as_saved,
    get_strategy_by_uid,
    get_internal_strategy_path,
    get_portfolios_for_uid,
    get_all_saved_uids,
    INTERNAL_STRATEGY_DIR,
)

# In-memory store for strategies loaded in this session (shared across pages)
# Key: strategy uid, Value: dict with metadata (uid, name, file_path, cached df, etc.)
p1_strategy_store = {}


def create_notification(level: str, text: str) -> dict:
    """
    Create a notification entry following the schema:
    {
      "id": "<uuid>",
      "timestamp": "<ISO-8601 datetime>",
      "level": "ERROR" | "SKIP",
      "text": "<full original message>"
    }
    """
    return {
        "id": uuid.uuid4().hex,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "text": text,
    }


def errors_to_notifications(errors: list) -> list:
    """
    Convert a list of error strings to notification entries.
    Parses [ERROR] and [SKIP] prefixes to determine level.
    """
    notifications = []
    for err in errors:
        if err.startswith("[ERROR]"):
            level = "ERROR"
        elif err.startswith("[SKIP]"):
            level = "SKIP"
        else:
            level = "SKIP"  # Default to SKIP for unknown prefixes
        notifications.append(create_notification(level, err))
    return notifications


OO_REQUIRED_COLUMNS = [
    "Date Opened",
    "Time Opened",
    "Opening Price",
    "Legs",
    "Premium",
    "Closing Price",
    "Date Closed",
    "Time Closed",
    "Avg. Closing Cost",
    "Reason For Close",
    "P/L",
]

# --------------------------------------------------------------------
# CONFIG: root data folder for folder-based loading
# --------------------------------------------------------------------
ROOT_DATA_DIR = r"C:\Users\mauro\MAURO\Options Trading\TradeBusters\Portfolio26\Phase1"

# --------------------------------------------------------------------
# CONFIG: app-owned folder for uploaded CSVs
# --------------------------------------------------------------------
# Base project directory: go one level up from /pages
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# App-owned folder for uploaded CSVs under core/dataupl
UPLOAD_DATA_DIR = os.path.join(
    BASE_DIR,
    "core",
    "dataupl",
)


os.makedirs(UPLOAD_DATA_DIR, exist_ok=True)


def _list_immediate_subfolders(root_path: str):
    """Return list of (folder_name, full_path) for direct subfolders of root_path."""
    if not os.path.isdir(root_path):
        return []
    items = []
    for name in sorted(os.listdir(root_path)):
        full_path = os.path.join(root_path, name)
        if os.path.isdir(full_path):
            items.append((name, full_path))
    return items

def build_data_input_section(folder_info, folder_checklist, portfolio_options):
    """
    Shared 'Data Input – Strategies' collapsible panel for Phase 1 / Phase 2 / etc.

    Parameters
    ----------
    folder_info : dash component
        Small info block about ROOT_DATA_DIR / folders.
    folder_checklist : dash component
        Checklist of subfolders to scan.
    portfolio_options : list[dict]
        Options for the portfolio dropdown.

    Returns
    -------
    dash component
        A dbc.Row containing the top controls + collapse with the full data input card.
    """

    data_input_card = dbc.Card(
        [
            dbc.CardHeader("Data Input – Strategies"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            # LEFT COLUMN: upload + folder loading
                            dbc.Col(
                                [
                                    # Upload section
                                    html.H6("Upload CSV files", className="mb-2"),
                                    html.P(
                                        "Drop one or more strategy CSV files here, or click to select files.",
                                        style={"fontSize": "0.85rem"},
                                    ),
                                    dcc.Upload(
                                        id="p1-upload",
                                        children=html.Div(
                                            "Drag & Drop or Click to Select CSV file(s)"
                                        ),
                                        multiple=True,
                                        style={
                                            "width": "100%",
                                            "height": "80px",
                                            "lineHeight": "80px",
                                            "borderWidth": "1px",
                                            "borderStyle": "dashed",
                                            "borderRadius": "4px",
                                            "textAlign": "center",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Div(
                                        id="p1-uploaded-files",
                                        style={"fontSize": "0.8rem"},
                                    ),
                                    html.Hr(),
                                    # Folder-based selection
                                    html.H6(
                                        "Load from data folders", className="mb-2"
                                    ),
                                    folder_info,
                                    folder_checklist,
                                    dbc.Button(
                                        "Scan selected folders for CSVs",
                                        id="p1-folder-scan-btn",
                                        n_clicks=0,
                                        color="primary",
                                        size="sm",
                                        className="mt-2",
                                    ),
                                    dbc.Button(
                                        "Load Selected Strategies",
                                        id="p1-load-strategies-btn",
                                        n_clicks=0,
                                        color="success",
                                        size="sm",
                                        className="mt-2",
                                        style={"marginLeft": "0.5rem"},
                                    ),
                                    html.Div(
                                        id="p1-folder-scan-result",
                                        style={
                                            "fontSize": "0.8rem",
                                            "marginTop": "0.5rem",
                                        },
                                    ),

                                ],
                                md=7,
                                lg=8,
                            ),
                            # RIGHT COLUMN: load existing portfolio
                            dbc.Col(
                                [
                                    html.H6(
                                        "Load existing portfolio", className="mb-2"
                                    ),
                                    html.P(
                                        "Select a saved portfolio to load its strategies.",
                                        style={"fontSize": "0.85rem"},
                                    ),
                                    dcc.Dropdown(
                                        id="p1-portfolio-dropdown",
                                        options=portfolio_options,
                                        value=None,
                                        placeholder=(
                                            "No portfolios saved yet"
                                            if not portfolio_options
                                            else "Select a portfolio."
                                        ),
                                        clearable=True,
                                        style={"fontSize": "0.85rem"},
                                    ),
                                    html.Div(
                                        id="p1-portfolio-summary",
                                        style={
                                            "fontSize": "0.8rem",
                                            "marginTop": "0.5rem",
                                        },
                                    ),
                                    dbc.Button(
                                        "Load Portfolio",
                                        id="p1-load-portfolio-btn",
                                        n_clicks=0,
                                        color="info",
                                        size="sm",
                                        className="mt-2",
                                    ),
                                ],
                                md=5,
                                lg=4,
                            ),
                        ]
                    )
                ]
            ),
        ]
    )

    # Wrap the controls row + collapse (same structure as in page1)
    return dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(
                        [
                            # Left side: controls
                            html.Div(
                                [
                                    dbc.Button(
                                        "Hide / Show Data Input",
                                        id="p1-toggle-data-panel",
                                        color="success",
                                        outline=True,
                                        size="sm",
                                    ),
                                    dcc.Input(
                                        id="p1-portfolio-name-input",
                                        type="text",
                                        placeholder="Portfolio name",
                                        style={
                                            "width": "220px",
                                            "fontSize": "0.85rem",
                                            "marginLeft": "0.75rem",
                                        },
                                    ),
                                    dbc.Button(
                                        "Save",
                                        id="p1-save-portfolio-btn",
                                        n_clicks=0,
                                        color="primary",
                                        size="sm",
                                        style={"marginLeft": "0.5rem"},
                                    ),
                                    dbc.Button(
                                        "Save As",
                                        id="p1-saveas-portfolio-btn",
                                        n_clicks=0,
                                        color="secondary",
                                        size="sm",
                                        style={"marginLeft": "0.25rem"},
                                    ),
                                    html.Div(
                                        id="p1-portfolio-save-status",
                                        style={
                                            "fontSize": "0.8rem",
                                            "marginLeft": "0.75rem",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "0.5rem",
                                    "flexWrap": "wrap",
                                },
                            ),
                            # Right side: Phase 1 composite score placeholder
                            html.Div(
                                id="p1-phase1-score-box",
                                children="Phase 1 score: N/A",
                                style={
                                    "marginLeft": "auto",
                                    "fontSize": "0.9rem",
                                    "padding": "0.25rem 0.75rem",
                                    "border": "1px solid #555555",
                                    "borderRadius": "4px",
                                    "backgroundColor": "#1f1f1f",
                                    "color": "#EEEEEE",
                                    "whiteSpace": "nowrap",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "space-between",
                            "gap": "0.75rem",
                            "marginBottom": "0.5rem",
                            "flexWrap": "wrap",
                        },
                    ),
                    dbc.Collapse(
                        id="p1-data-panel",
                        is_open=False,
                        children=[data_input_card],
                    ),
                ],
                width=12,
            )
        ]
    )


def build_strategy_sidebar():
    """
    Shared left-hand 'Strategy List' sidebar (Phase 1 / Phase 2 / etc).
    
    Now split into two panels:
    - Active List (top): strategies currently in the working set (is_active=True)
    - Universe List (bottom): all saved strategies (is_saved=True)
    """
    return dbc.Col(
        [
            # Stores for strategy states
            dcc.Store(id="p1-strategy-registry-sync"),
            dcc.Store(id="p1-active-list-store", data=[]),
            dcc.Store(id="p1-universe-list-store", data=[]),
            
            # Hidden checklist to maintain backward compatibility with existing callbacks
            html.Div(
                dcc.Checklist(
                    id="p1-strategy-checklist",
                    options=[],
                    value=[],
                ),
                style={"display": "none"},
            ),
            
            # ===== Active List (top block) =====
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.Span(
                                "Active",
                                style={"fontWeight": "bold"},
                            ),
                            html.Span(
                                id="p1-active-count-badge",
                                children=" (0)",
                                style={"fontSize": "0.85rem", "color": "#AAAAAA"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"},
                    ),
                    dbc.CardBody(
                        [
                            # Select/Deselect all for Active list
                            dbc.Checkbox(
                                id="p1-strategy-select-all",
                                value=False,
                                label="Select / deselect all",
                                style={
                                    "fontSize": "0.8rem",
                                    "marginBottom": "0.4rem",
                                },
                            ),
                            # Active list container - rendered dynamically
                            html.Div(
                                id="p1-active-list-container",
                                children=[
                                    html.Div(
                                        "No active strategies.",
                                        style={"fontSize": "0.8rem", "color": "#888888"},
                                    )
                                ],
                                style={
                                    "maxHeight": "200px",
                                    "overflowY": "auto",
                                    "fontSize": "0.85rem",
                                },
                            ),
                        ],
                        style={"paddingTop": "0.5rem", "paddingBottom": "0.5rem"},
                    ),
                ],
                style={"marginBottom": "0.5rem"},
            ),
            
            # ===== Universe List (bottom block) =====
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.Span(
                                "Universe",
                                style={"fontWeight": "bold"},
                            ),
                            html.Span(
                                id="p1-universe-count-badge",
                                children=" (0)",
                                style={"fontSize": "0.85rem", "color": "#AAAAAA"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"},
                    ),
                    dbc.CardBody(
                        [
                            # Search box
                            dcc.Input(
                                id="p1-universe-search",
                                type="text",
                                placeholder="Search strategies...",
                                style={
                                    "width": "100%",
                                    "fontSize": "0.8rem",
                                    "marginBottom": "0.4rem",
                                    "padding": "0.25rem 0.5rem",
                                    "borderRadius": "4px",
                                    "border": "1px solid #444444",
                                    "backgroundColor": "#2a2a2a",
                                    "color": "#EEEEEE",
                                },
                            ),
                            # Header controls
                            html.Div(
                                [
                                    dbc.Checkbox(
                                        id="p1-universe-select-all",
                                        value=False,
                                        label="Select all (filtered)",
                                        style={"fontSize": "0.75rem"},
                                    ),
                                    dbc.Button(
                                        "Move → Active",
                                        id="p1-universe-move-to-active-btn",
                                        color="primary",
                                        size="sm",
                                        outline=True,
                                        style={"fontSize": "0.7rem", "padding": "0.15rem 0.4rem"},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center",
                                    "marginBottom": "0.4rem",
                                },
                            ),
                            # Universe list container - rendered dynamically
                            html.Div(
                                id="p1-universe-list-container",
                                children=[
                                    html.Div(
                                        "No strategies in universe.",
                                        style={"fontSize": "0.8rem", "color": "#888888"},
                                    )
                                ],
                                style={
                                    "maxHeight": "250px",
                                    "overflowY": "auto",
                                    "fontSize": "0.85rem",
                                },
                            ),
                        ],
                        style={"paddingTop": "0.5rem", "paddingBottom": "0.5rem"},
                    ),
                ],
            ),
            
            # Summary message under both lists
            html.Div(
                id="p1-strategy-summary",
                children="",
                style={
                    "fontSize": "0.8rem",
                    "marginTop": "0.4rem",
                    "color": "#AAAAAA",
                },
            ),
        ],
        width=3,
        style={"position": "sticky", "top": "80px"},
    )

@callback(
    Output("p1-uploaded-files", "children"),
    Input("p1-upload", "filename"),
)
def show_uploaded_files(filenames):
    """
    Display a list of uploaded CSV filenames (single or multiple).

    NOTE: At this stage we ONLY show the names. Actual parsing and
    registry building will be added in the next step.
    """
    if not filenames:
        return ""

    # Dash sends filename as str for single file, list for multiple
    if isinstance(filenames, str):
        filenames = [filenames]

    items = [html.Li(name) for name in filenames]
    return html.Div(
        [
            html.Div("Uploaded file(s):", style={"fontWeight": "bold"}),
            html.Ul(items),
        ]
    )


@callback(
    Output("p1-folder-scan-result", "children"),
    Input("p1-folder-scan-btn", "n_clicks"),
    State("p1-folder-checklist", "value"),
)
def scan_selected_folders_for_csv(n_clicks, selected_folders):
    """
    When user clicks 'Scan selected folders', list all CSV files found
    in the selected subfolders of ROOT_DATA_DIR.
    """
    if not n_clicks:
        return ""

    if not selected_folders:
        return html.Div(
            "No folders selected.", style={"color": "orange", "fontSize": "0.85rem"}
        )

    all_results = []
    total_files = 0

    for folder in selected_folders:
        if not os.path.isdir(folder):
            all_results.append(
                html.Div(
                    f"[WARNING] Folder does not exist: {folder}",
                    style={"color": "red", "fontSize": "0.8rem"},
                )
            )
            continue

        try:
            files = os.listdir(folder)
        except Exception as e:
            all_results.append(
                html.Div(
                    f"[ERROR] Cannot read folder {folder}: {e}",
                    style={"color": "red", "fontSize": "0.8rem"},
                )
            )
            continue

        csv_files = sorted([f for f in files if f.lower().endswith(".csv")])

        total_files += len(csv_files)
        if csv_files:
            all_results.append(
                html.Div(
                    [
                        html.Div(
                            os.path.basename(folder),
                            style={"fontWeight": "bold", "marginTop": "0.4rem"},
                        ),
                        html.Ul([html.Li(f) for f in csv_files]),
                    ]
                )
            )
        else:
            all_results.append(
                html.Div(
                    [
                        html.Div(
                            os.path.basename(folder),
                            style={"fontWeight": "bold", "marginTop": "0.4rem"},
                        ),
                        html.Div(
                            "No CSV files found in this folder.",
                            style={"fontSize": "0.8rem"},
                        ),
                    ]
                )
            )

    summary = html.Div(
        f"Total CSV files found across selected folders: {total_files}",
        style={"fontWeight": "bold", "marginBottom": "0.3rem"},
    )
    return html.Div([summary] + all_results)


@callback(
    Output("p1-data-panel", "is_open"),
    Input("p1-toggle-data-panel", "n_clicks"),
    State("p1-data-panel", "is_open"),
)
def toggle_data_panel(n_clicks, is_open):
    """
    Toggle the visibility of the Data Input panel (collapse/expand).
    """
    if not n_clicks:
        return is_open
    return not is_open

@callback(
    Output("p1-strategy-registry-sync", "data"),
    Input("p1-strategy-checklist", "value"),
)
def sync_registry_phase1_active(selected_ids):
    """
    Update `phase1_active` flags in the JSON registry based on the current
    checklist selection. Returns a small summary payload just to satisfy Dash.
    """
    selected_ids = selected_ids or []

    # Update JSON registry
    set_phase1_active_flags(selected_ids)

    # Read back for a quick summary
    registry = load_registry()
    n_total = len(registry.get("strategies", []))

    return {
        "n_selected": len(selected_ids),
        "n_total": n_total,
    }

@callback(
    Output("p1-portfolio-save-status", "children"),
    Output("p1-current-portfolio-id", "data", allow_duplicate=True),
    Output("p1-portfolio-dropdown", "options"),
    Output("p1-portfolio-dropdown", "value"),
    Input("p1-save-portfolio-btn", "n_clicks"),
    Input("p1-saveas-portfolio-btn", "n_clicks"),
    State("p1-current-portfolio-id", "data"),
    State("p1-strategy-checklist", "value"),
    State("p1-portfolio-name-input", "value"),
    prevent_initial_call=True,
)
def save_or_saveas_portfolio(
    n_click_save,
    n_click_save_as,
    current_portfolio_id,
    selected_strategy_ids,
    portfolio_name,
):
    """
    Save / Save As logic for Phase 1 portfolios.

    Rules:
    - SAVE:
        * Only valid if current_portfolio_id is not None.
        * Updates that existing portfolio with current selection.
        * NEVER changes the portfolio name (ignores the name input).
    - SAVE AS:
        * Always creates a new portfolio id, using the entered name.
        * Enforces uniqueness of the portfolio name.
    - Registry is ONLY modified here (add/update, never delete).
    """
    triggered_id = ctx.triggered_id

    # Normalise
    selected_strategy_ids = selected_strategy_ids or []
    name_clean = (portfolio_name or "").strip()

    # Helper: rebuild portfolio dropdown from registry
    def _build_portfolio_options_and_value(selected_id):
        """
        Build dropdown options + current value from the *latest* registry.
    
        We read via load_registry() instead of list_portfolios() to avoid
        any stale in-memory cache and always reflect the most recent save.
        """
        registry = load_registry() or {}
        portfolios = registry.get("portfolios", [])
    
        options = [
            {
                "label": p.get("name", p.get("id", "UNKNOWN")),
                "value": p["id"],
            }
            for p in sorted(
                portfolios,
                key=lambda p: p.get("name", p.get("id", "")),
            )
            if "id" in p
        ]
    
        return options, selected_id


    # No strategies selected -> nothing to save
    if not selected_strategy_ids:
        status = "No strategies selected. Portfolio not saved."
        options, value = _build_portfolio_options_and_value(current_portfolio_id)
        return status, current_portfolio_id, options, value

    # Decide which action
    if triggered_id == "p1-saveas-portfolio-btn":
        action = "saveas"
    elif triggered_id == "p1-save-portfolio-btn":
        action = "save"
    else:
        status = "No save action triggered."
        options, value = _build_portfolio_options_and_value(current_portfolio_id)
        return status, current_portfolio_id, options, value

    # ---------- Basic rules for SAVE vs SAVE AS ----------

    # SAVE: must have an existing portfolio id; name input is ignored
    if action == "save":
        if not current_portfolio_id:
            status = (
                "Cannot use 'Save' because no existing portfolio is loaded. "
                "Use 'Save As' to create a new portfolio."
            )
            options, value = _build_portfolio_options_and_value(current_portfolio_id)
            return status, current_portfolio_id, options, value

    # SAVE AS: must have a non-empty name and must be unique
    else:  # saveas
        if not name_clean:
            status = "Please enter a portfolio name before using 'Save As'."
            options, value = _build_portfolio_options_and_value(current_portfolio_id)
            return status, current_portfolio_id, options, value

        # Enforce unique name (case-insensitive)
        portfolios = list_portfolios()
        existing_names = {
            (p.get("name") or "").strip().lower() for p in portfolios
        }
        if name_clean.lower() in existing_names:
            status = (
                f"Portfolio '{name_clean}' already exists. "
                "Use a different name or 'Save' to update it."
            )
            options, value = _build_portfolio_options_and_value(current_portfolio_id)
            return status, current_portfolio_id, options, value

    # ---------- Determine target portfolio id and name ----------
    from datetime import datetime

    if action == "save":
        # Update existing portfolio; DO NOT change its name
        existing = get_portfolio(current_portfolio_id)
        if existing is None:
            status = (
                "Existing portfolio not found in registry. "
                "Use 'Save As' to create a new one."
            )
            options, value = _build_portfolio_options_and_value(current_portfolio_id)
            return status, current_portfolio_id, options, value

        portfolio_id = existing["id"]
        portfolio_name_final = existing.get("name", "Unnamed Phase 1 portfolio")

    else:
        # Create a new portfolio id based on the name + timestamp
        portfolio_name_final = name_clean
        portfolio_id = f"ph1::{portfolio_name_final}::{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # ---------- Ensure all selected strategies exist/updated in registry ----------
    # Collect UIDs for the portfolio and source paths for copying
    selected_uids = []
    source_paths = {}
    
    for sid in selected_strategy_ids:
        # Start from whatever we have in the in-memory store
        meta = (p1_strategy_store.get(sid) or {}).copy()

        # Always enforce the id (legacy compatibility)
        meta["id"] = sid

        # file_path: if missing, default to the id (for folder-based strategies)
        if not meta.get("file_path"):
            meta["file_path"] = sid

        # Derive uid from file_path
        uid = derive_uid_from_filepath(meta["file_path"])
        meta["uid"] = uid
        meta["file_name"] = f"{uid}.csv"
        selected_uids.append(uid)
        
        # Track source path for copying
        source_paths[uid] = meta["file_path"]

        # name: if missing, derive from the filename (without .csv)
        if not meta.get("name"):
            base = os.path.basename(sid)
            if base.lower().endswith(".csv"):
                base = base[:-4]
            meta["name"] = base or sid

        # folder: if missing, derive from parent directory
        if not meta.get("folder"):
            parent = os.path.dirname(sid)
            if parent:
                meta["folder"] = os.path.basename(parent)
            else:
                meta["folder"] = "(unknown)"

        # source: if missing, guess from path
        if not meta.get("source"):
            try:
                norm = os.path.normpath(sid).lower()
                if "dataupl" in norm:
                    meta["source"] = "upload"
                else:
                    meta["source"] = "folder"
            except Exception:
                meta["source"] = "folder"

        # n_rows: if missing, try from cached df
        if not meta.get("n_rows"):
            n_rows = 0
            df_cached = meta.get("df")
            if df_cached is not None:
                try:
                    n_rows = len(df_cached)
                except Exception:
                    n_rows = 0
            meta["n_rows"] = n_rows

        # Keep store and registry in sync (use uid as key now)
        p1_strategy_store[sid] = meta
        p1_strategy_store[uid] = meta  # Also store by uid
        add_or_update_strategy(meta)

    # ---------- Copy strategies to internal storage if not already saved ----------
    success_uids, failed_uids = mark_strategies_as_saved(selected_uids, source_paths)
    
    # Build warning message for failed copies
    copy_warning = ""
    if failed_uids:
        copy_warning = f" Warning: {len(failed_uids)} strategy(s) could not be saved internally."

    # ---------- Add or update portfolio in registry ----------
    portfolio_obj = {
        "id": portfolio_id,
        "name": portfolio_name_final,
        "strategy_uids": selected_uids,
        "strategy_ids": selected_strategy_ids,  # Keep for backward compatibility
        "phase1_done": True,
    }
    add_or_update_portfolio(portfolio_obj)

    # Refresh dropdown and select current portfolio
    options, value = _build_portfolio_options_and_value(portfolio_id)

    if action == "save":
        action_label = "updated"
    else:
        action_label = "created"

    status = (
        f"Portfolio '{portfolio_name_final}' {action_label} with "
        f"{len(selected_strategy_ids)} strategies.{copy_warning}"
    )

    # current_portfolio_id becomes portfolio_id after any successful save
    return status, portfolio_id, options, value


@callback(
    Output("p1-strategy-checklist", "options"),
    Output("p1-strategy-checklist", "value"),
    Output("p1-strategy-select-all", "value"),
    Output("p1-strategy-summary", "children"),
    Output("p1-current-portfolio-id", "data"),
    Output("app-notifications", "data"),
    Input("p1-load-strategies-btn", "n_clicks"),
    Input("p1-load-portfolio-btn", "n_clicks"),
    Input("p1-strategy-select-all", "value"),
    State("p1-folder-checklist", "value"),
    State("p1-upload", "contents"),
    State("p1-upload", "filename"),
    State("p1-strategy-checklist", "options"),
    State("p1-strategy-checklist", "value"),
    State("p1-portfolio-dropdown", "value"),
    State("p1-current-portfolio-id", "data"),
    State("app-notifications", "data"),
)
def load_or_toggle_strategies(
    n_clicks_load,
    n_clicks_load_portfolio,
    select_all_value,
    selected_folders,
    upload_contents,
    upload_filenames,
    existing_options,
    existing_values,
    selected_portfolio_id,
    current_portfolio_id,
    existing_notifications,
):
    """
    Dispatcher for:
      - Loading strategies from CSVs (Load Selected Strategies)
      - Loading an existing portfolio
      - Selecting/deselecting all

    IMPORTANT:
    - This callback never clears the registry.
    - CSV loads merge into the current list; they do not replace it.
    - Errors are routed to the app-notifications store instead of inline display.
    """
    triggered_id = ctx.triggered_id
    
    # Initialize existing notifications if None
    existing_notifications = existing_notifications or []

    # Initial state: nothing has happened yet
    if triggered_id is None:
        return [], [], False, "No strategies loaded yet.", current_portfolio_id, existing_notifications


    existing_options = existing_options or []
    existing_values = existing_values or []
    
    # Load registry once so it is available to all CASE branches
    registry = load_registry()


    # ----------------------------------------------------------
    # CASE 1: Load button clicked -> add CSV strategies IN MEMORY
    # ----------------------------------------------------------
    if triggered_id == "p1-load-strategies-btn":
        errors = []

        # If this is a completely fresh session (no options, no portfolio),
        # reset the in-memory store; otherwise we append to it.
        if not existing_options and not current_portfolio_id:
            p1_strategy_store.clear()

        # Build a dict from existing options: id -> (id, label)
        option_map = {
            opt["value"]: (opt["value"], opt["label"]) for opt in existing_options
        }

        new_ids = []

        # ---------- FROM FOLDERS ----------
        if selected_folders:
            # First, collect all candidate UIDs to validate uniqueness
            registry = load_registry()
            
            for folder in selected_folders:
                if not os.path.isdir(folder):
                    errors.append(f"[ERROR] Folder does not exist: {folder}")
                    continue

                try:
                    files = os.listdir(folder)
                except Exception as e:
                    errors.append(f"[ERROR] Cannot read folder {folder}: {e}")
                    continue

                for f in files:
                    if not f.lower().endswith(".csv"):
                        continue

                    full_path = os.path.join(folder, f)
                    
                    # Derive UID and check for uniqueness
                    uid = derive_uid_from_filepath(full_path)
                    
                    # Check if UID already exists in registry or in current batch
                    if uid_exists_in_registry(uid, registry) or uid in [
                        derive_uid_from_filepath(opt["value"]) 
                        for opt in existing_options
                    ]:
                        errors.append(
                            f"[SKIP] Strategy '{uid}' already exists in registry. "
                            f"Skipping duplicate from {full_path}."
                        )
                        continue

                    try:
                        df = pd.read_csv(full_path)
                    except Exception as e:
                        errors.append(f"[ERROR] Failed to read {full_path}: {e}")
                        continue
                    
                    # OO log structure check
                    missing_cols = [c for c in OO_REQUIRED_COLUMNS if c not in df.columns]
                    if missing_cols:
                        errors.append(
                            f"[SKIP] {full_path} is not a valid OO log file "
                            f"(missing required columns: {', '.join(missing_cols)})."
                        )
                        continue
                    
                    n_rows = len(df)
                    
                    strategy_id = full_path
                    # Decide which columns to cache: always Date Closed + P/L,
                    # plus any extra parameter columns when available.
                    cache_cols = ["Date Closed", "P/L"]
                    for extra_col in [
                        "Margin Req.",
                        "Date Opened",
                        "Time Opened",
                        "Premium",
                        "Gap",
                        "Movement",
                    ]:
                        if extra_col in df.columns:
                            cache_cols.append(extra_col)
                    
                    strategy_meta = {
                        "id": strategy_id,
                        "uid": uid,
                        "file_name": f"{uid}.csv",
                        "is_saved": False,
                        "name": f"FOLD. {os.path.splitext(f)[0]}",
                        "file_path": full_path,
                        "folder": os.path.basename(folder),
                        "source": "folder",
                        "n_rows": n_rows,
                        # Cache a trimmed DataFrame for analytics (including Margin Req. and
                        # basic parameter columns when available)
                        "df": df[cache_cols].copy(),
                    }

                    p1_strategy_store[strategy_id] = strategy_meta
                    p1_strategy_store[uid] = strategy_meta  # Also store by uid

                    if strategy_id not in option_map:
                        option_map[strategy_id] = (
                            strategy_id,
                            strategy_meta["name"],
                        )
                        new_ids.append(strategy_id)

        # ---------- FROM UPLOADS ----------
        if upload_contents and upload_filenames:
            if isinstance(upload_contents, str):
                upload_contents = [upload_contents]
            if isinstance(upload_filenames, str):
                upload_filenames = [upload_filenames]
            
            # Load registry once for UID checks if not loaded during folder processing
            # if 'registry' not in locals():
            #     registry = load_registry()
        
            for content, fname in zip(upload_contents, upload_filenames):
                try:
                    content_type, content_string = content.split(",", 1)
                    decoded = base64.b64decode(content_string)
                    s = decoded.decode("utf-8")
                    df = pd.read_csv(io.StringIO(s))
                except Exception as e:
                    errors.append(f"[ERROR] Failed to read uploaded file {fname}: {e}")
                    continue
        
                # OO log structure check
                missing_cols = [c for c in OO_REQUIRED_COLUMNS if c not in df.columns]
                if missing_cols:
                    errors.append(
                        f"[SKIP] uploaded file {fname} is not a valid OO log file "
                        f"(missing required columns: {', '.join(missing_cols)})."
                    )
                    continue
        
                n_rows = len(df)
        
                # ------------------------------------------------------
                # Persist uploaded file under core/dataupl so that
                # it behaves exactly like a folder-based strategy.
                # ------------------------------------------------------
                try:
                    # Make sure the directory exists (idempotent)
                    os.makedirs(UPLOAD_DATA_DIR, exist_ok=True)
        
                    base_name = os.path.basename(fname)
                    name_no_ext, ext = os.path.splitext(base_name)
                    # Sanitize filename to avoid weird characters
                    safe_base = "".join(
                        ch if ch.isalnum() or ch in ("-", "_") else "_"
                        for ch in name_no_ext
                    )
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    stored_filename = f"{safe_base}_{timestamp}{ext or '.csv'}"
                    stored_path = os.path.join(UPLOAD_DATA_DIR, stored_filename)
        
                    # Save the full dataframe to disk
                    df.to_csv(stored_path, index=False)
                except Exception as e:
                    errors.append(
                        f"[ERROR] Failed to save uploaded file {fname} to data folder: {e}"
                    )
                    continue
                
                # Derive UID from stored path and check for uniqueness
                uid = derive_uid_from_filepath(stored_path)
                
                # Check if UID already exists in registry or in current batch
                if uid_exists_in_registry(uid, registry) or uid in [
                    derive_uid_from_filepath(opt["value"]) 
                    for opt in existing_options
                ]:
                    errors.append(
                        f"[SKIP] Strategy '{uid}' already exists in registry. "
                        f"Skipping duplicate upload {fname}."
                    )
                    # Remove the stored file since we're skipping this upload
                    try:
                        os.remove(stored_path)
                    except Exception:
                        pass
                    continue
        
                # Use the stored path as the strategy id (same convention as folder-based)
                strategy_id = stored_path
                # Decide which columns to cache: always Date Closed + P/L, plus Margin Req. if present
                cache_cols = ["Date Closed", "P/L"]
                if "Margin Req." in df.columns:
                    cache_cols.append("Margin Req.")
                
                strategy_meta = {
                    "id": strategy_id,
                    "uid": uid,
                    "file_name": f"{uid}.csv",
                    "is_saved": False,
                    "name": f"SINGLE. {os.path.splitext(fname)[0]}",
                    "file_path": stored_path,
                    "folder": "dataupl",
                    "source": "upload",
                    "n_rows": n_rows,
                    # Cache a trimmed DataFrame for analytics (including Margin Req. when available)
                    "df": df[cache_cols].copy(),
                }

        
                p1_strategy_store[strategy_id] = strategy_meta
                p1_strategy_store[uid] = strategy_meta  # Also store by uid
        
                if strategy_id not in option_map:
                    option_map[strategy_id] = (
                        strategy_id,
                        strategy_meta["name"],
                    )
                    new_ids.append(strategy_id)


        # If we still have nothing, report and keep context
        if not option_map:
            summary = "No strategies loaded (no valid CSVs found)."
            # Route errors to notifications instead of inline display
            new_notifications = errors_to_notifications(errors) if errors else []
            updated_notifications = existing_notifications + new_notifications
            return [], [], False, summary, current_portfolio_id, updated_notifications

        # Build options sorted by label
        id_label_pairs = list(option_map.values())
        id_label_pairs.sort(key=lambda x: x[1])

        options = [
            {"label": label, "value": sid} for sid, label in id_label_pairs
        ]

        # Keep existing selections and add new ones as selected
        selected_set = set(existing_values)
        selected_set.update(new_ids)
        selected_values = list(selected_set)

        select_all_flag = len(selected_values) == len(options)
       
        if new_ids:
            base_summary = (
                f"[INFO] Added {len(new_ids)} new strategies from folders/uploads. "
                f"{len(selected_values)} of {len(options)} strategies selected for Phase 1."
            )
        else:
            base_summary = (
                f"[INFO] No new strategies added. "
                f"{len(selected_values)} of {len(options)} strategies selected for Phase 1."
            )
        
        # Push this into the same pipeline that already creates notifications
        errors.append(base_summary)


        # Route errors to notifications instead of inline display
        new_notifications = errors_to_notifications(errors) if errors else []
        updated_notifications = existing_notifications + new_notifications
                   
        # We no longer display any status text under the Strategy List;
        # all feedback goes through the notification panel.
        status_message = ""

        # IMPORTANT: keep current_portfolio_id unchanged
        return options, selected_values, select_all_flag, status_message, current_portfolio_id, updated_notifications

    # ----------------------------------------------------------
    # CASE 2: Load existing portfolio (from registry)
    # ----------------------------------------------------------
    if triggered_id == "p1-load-portfolio-btn":
        options = existing_options or []
    
        if not selected_portfolio_id:
            summary = "No portfolio selected."
            current_values = []
            return options, current_values, False, summary, None, existing_notifications
    
        portfolio = get_portfolio(selected_portfolio_id)
        if not portfolio:
            summary = f"Portfolio '{selected_portfolio_id}' not found in registry."
            current_values = []
            return options, current_values, False, summary, None, existing_notifications
    
        # Get portfolio strategy UIDs (prefer strategy_uids, fallback to strategy_ids)
        portfolio_uids = set(
            portfolio.get("strategy_uids", portfolio.get("strategy_ids", []))
        )
    
        # All strategies currently known in the registry
        all_strats = list_strategies()
        if not all_strats:
            summary = (
                "Registry has no strategies. "
                "Save at least one portfolio before using this feature."
            )
            return [], [], False, summary, selected_portfolio_id, existing_notifications
    
        # Build checklist options **only** from strategies in this portfolio
        options = []
        selected_values = []
    
        for s in all_strats:
            uid = s.get("uid")
            sid = s.get("id")
    
            # Check if this strategy is in the portfolio (by uid or id for backward compat)
            in_portfolio = (uid in portfolio_uids) or (sid in portfolio_uids)
            if not in_portfolio:
                continue
    
            label = s.get("name", uid or sid or "UNKNOWN")
            options.append(
                {
                    "label": label,
                    "value": sid,
                }
            )
            selected_values.append(sid)
    
            # Load strategy data from internal storage if is_saved=True
            if s.get("is_saved", False) and uid:
                internal_path = get_internal_strategy_path(uid)
                if internal_path.exists():
                    try:
                        df = pd.read_csv(internal_path)
    
                        # Cache required columns
                        cache_cols = ["Date Closed", "P/L"]
                        for extra_col in [
                            "Margin Req.",
                            "Date Opened",
                            "Time Opened",
                            "Premium",
                            "Gap",
                            "Movement",
                        ]:
                            if extra_col in df.columns:
                                cache_cols.append(extra_col)
    
                        strategy_meta = dict(s)
                        strategy_meta["df"] = df[cache_cols].copy()
                        strategy_meta["file_path"] = str(internal_path)
    
                        # Store in memory by both id and uid
                        p1_strategy_store[sid] = strategy_meta
                        p1_strategy_store[uid] = strategy_meta
                    except (FileNotFoundError, PermissionError, pd.errors.EmptyDataError):
                        # Strategy file missing or inaccessible - continue with registry data
                        pass
                    except Exception:
                        # Unexpected error loading strategy - continue with registry data
                        pass
    
        # Count missing strategies: portfolio_uids that did not match any strategy
        n_found = len(selected_values)
        n_expected = len(portfolio_uids)
        n_missing = n_expected - n_found
    
        # Get universe count (all saved strategies)
        all_saved_uids = get_all_saved_uids(registry)
        n_universe = len(all_saved_uids)
    
        select_all_flag = len(selected_values) == len(options)
    
        # Summary message format: "Loaded portfolio 'X' – Y active out of Z in universe."
        summary = (
            f"Loaded portfolio '{portfolio.get('name', selected_portfolio_id)}' – "
            f"{len(selected_values)} active out of {n_universe} in universe."
        )
        if n_missing > 0:
            summary += f" ({n_missing} uid(s) not found.)"
    
        # Portfolio context is set to the loaded portfolio id
        return options, selected_values, select_all_flag, summary, selected_portfolio_id, existing_notifications




    # ----------------------------------------------------------
    # CASE 3: Select-all checkbox toggled
    # ----------------------------------------------------------
    if triggered_id == "p1-strategy-select-all":
        options = existing_options or []
        if not options:
            return [], [], False, "", current_portfolio_id, existing_notifications

        if select_all_value:
            values = [opt["value"] for opt in options]
        else:
            values = []

        # Summary shows selected count
        summary = f"Selected {len(values)} of {len(options)} active strategies."
        return options, values, select_all_value, summary, current_portfolio_id, existing_notifications

    # Fallback (shouldn't hit here)
    options = existing_options or []
    if not options:
        return [], [], False, "", current_portfolio_id, existing_notifications
    return options, [], False, "", current_portfolio_id, existing_notifications


# ===========================================================================
# Active / Universe Lists – New Callbacks
# ===========================================================================

def get_portfolio_badge_color(count: int) -> str:
    """
    Return badge color based on portfolio membership count.
    Grey for 0, Blue→Green→Yellow→Red gradient for 1, 2, 3-4, 5, 6+
    """
    if count == 0:
        return "#666666"
    elif count == 1:
        return "#1f77b4"  # Blue
    elif count == 2:
        return "#2ca02c"  # Green
    elif count <= 4:
        return "#ffbb00"  # Yellow
    elif count == 5:
        return "#ff7f0e"  # Orange
    else:
        return "#d62728"  # Red


def _build_active_row(uid: str, name: str, is_selected: bool, is_saved: bool, portfolio_count: int = 0):
    """
    Build a single row for the Active List with:
    - Checkbox for is_selected
    - Strategy name
    - Remove (×) icon
    - Saved shading background
    - Optional portfolio badge
    """
    row_style = {
        "display": "flex",
        "alignItems": "center",
        "padding": "0.25rem 0.5rem",
        "marginBottom": "0.15rem",
        "borderRadius": "4px",
    }
    
    # Add tinted background for saved strategies
    if is_saved:
        row_style["backgroundColor"] = "rgba(31, 119, 180, 0.15)"  # subtle blue tint
    
    return html.Div(
        [
            # Checkbox for is_selected
            dbc.Checkbox(
                id={"type": "active-row-checkbox", "uid": uid},
                value=is_selected,
                style={"marginRight": "0.5rem"},
            ),
            # Strategy name
            html.Span(
                name,
                style={
                    "flex": "1",
                    "fontSize": "0.8rem",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap",
                },
                title=name,
            ),
            # Portfolio badge (optional)
            html.Span(
                str(portfolio_count) if portfolio_count > 0 else "",
                style={
                    "fontSize": "0.65rem",
                    "padding": "0.1rem 0.3rem",
                    "borderRadius": "8px",
                    "backgroundColor": get_portfolio_badge_color(portfolio_count),
                    "color": "white",
                    "marginRight": "0.3rem",
                    "display": "inline-block" if is_saved and portfolio_count > 0 else "none",
                },
                title=f"In {portfolio_count} portfolio(s)" if is_saved else "",
            ),
            # Remove button
            html.Button(
                "×",
                id={"type": "active-row-remove", "uid": uid},
                style={
                    "border": "none",
                    "background": "none",
                    "color": "#ff6666",
                    "cursor": "pointer",
                    "fontSize": "1rem",
                    "padding": "0 0.25rem",
                    "lineHeight": "1",
                },
                title="Remove from Active",
            ),
        ],
        style=row_style,
    )


def _build_universe_row(uid: str, name: str, is_multi_selected: bool, is_active: bool, 
                        portfolio_count: int = 0, portfolio_names: list = None):
    """
    Build a single row for the Universe List with:
    - Checkbox for multi-select
    - Strategy name
    - Quick-add icon
    - Portfolio badge with hover tooltip
    """
    portfolio_names = portfolio_names or []
    tooltip_text = ", ".join(portfolio_names) if portfolio_names else "Not in any portfolio"
    
    row_style = {
        "display": "flex",
        "alignItems": "center",
        "padding": "0.25rem 0.5rem",
        "marginBottom": "0.15rem",
        "borderRadius": "4px",
    }
    
    # Dim the row if already active
    if is_active:
        row_style["opacity"] = "0.5"
    
    return html.Div(
        [
            # Checkbox for multi-select
            dbc.Checkbox(
                id={"type": "universe-row-checkbox", "uid": uid},
                value=is_multi_selected,
                style={"marginRight": "0.5rem"},
                disabled=is_active,  # Disable if already active
            ),
            # Strategy name
            html.Span(
                name,
                style={
                    "flex": "1",
                    "fontSize": "0.8rem",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap",
                },
                title=name,
            ),
            # Portfolio badge
            html.Span(
                str(portfolio_count),
                style={
                    "fontSize": "0.65rem",
                    "padding": "0.1rem 0.3rem",
                    "borderRadius": "8px",
                    "backgroundColor": get_portfolio_badge_color(portfolio_count),
                    "color": "white" if portfolio_count > 0 else "#AAAAAA",
                    "marginRight": "0.3rem",
                },
                title=tooltip_text,
            ),
            # Quick-add button
            html.Button(
                "+",
                id={"type": "universe-row-quickadd", "uid": uid},
                style={
                    "border": "none",
                    "background": "none",
                    "color": "#66ff66" if not is_active else "#666666",
                    "cursor": "pointer" if not is_active else "not-allowed",
                    "fontSize": "1rem",
                    "padding": "0 0.25rem",
                    "lineHeight": "1",
                },
                title="Add to Active" if not is_active else "Already in Active",
                disabled=is_active,
            ),
        ],
        style=row_style,
    )


@callback(
    Output("p1-active-list-container", "children"),
    Output("p1-active-count-badge", "children"),
    Output("p1-active-list-store", "data"),
    Output("p1-strategy-checklist", "value", allow_duplicate=True),
    Input("p1-strategy-checklist", "options"),
    Input("p1-strategy-checklist", "value"),
    Input({"type": "active-row-checkbox", "uid": ALL}, "value"),
    Input({"type": "active-row-remove", "uid": ALL}, "n_clicks"),
    State("p1-active-list-store", "data"),
    prevent_initial_call=True,
)
def update_active_list_display(
    checklist_options,
    checklist_values,
    checkbox_values,
    remove_clicks,
    active_store,
):
    """
    Render the Active List based on current strategy state.

    Active List shows strategies present in p1-strategy-checklist.options.
    - Checkbox toggles is_selected (and updates checklist values).
    - Remove button is handled by remove_from_active_list; this callback
      just reflects the current contents of options/value.
    """
    triggered_id = ctx.triggered_id

    # Normalise inputs
    active_store = active_store or []
    checklist_options = checklist_options or []
    checklist_values = checklist_values or []
    checkbox_values = checkbox_values or []

    registry = load_registry()

    # Build base list of active strategies from options + current checklist values
    active_strategies = []
    for opt in checklist_options:
        sid = opt.get("value")
        label = opt.get("label", "")

        # Default selection state from checklist values
        is_selected = sid in checklist_values

        # Strategy metadata
        meta = p1_strategy_store.get(sid, {})
        uid = meta.get("uid") or derive_uid_from_filepath(sid)
        name = meta.get("name") or label or uid
        is_saved = meta.get("is_saved", False)

        # Portfolio count for saved strategies
        portfolio_count = 0
        if is_saved:
            portfolios = get_portfolios_for_uid(uid, registry)
            portfolio_count = len(portfolios)

        active_strategies.append(
            {
                "uid": uid,
                "sid": sid,
                "name": name,
                "is_selected": is_selected,
                "is_saved": is_saved,
                "portfolio_count": portfolio_count,
            }
        )

    # Single source of truth for selection:
    # - If a row checkbox was clicked, recompute selection from checkbox_values.
    # - Otherwise, keep checklist_values as-is.
    new_values = list(checklist_values)

    if isinstance(triggered_id, dict) and triggered_id.get("type") == "active-row-checkbox":
        new_values = []
        for idx, s in enumerate(active_strategies):
            # checkbox_values is aligned with rows by index
            selected = False
            if idx < len(checkbox_values) and checkbox_values[idx]:
                selected = True
            s["is_selected"] = selected
            if selected:
                new_values.append(s["sid"])

        # Keep checklist_values in sync for subsequent renders
        checklist_values = new_values

    # Build rows for display
    if not active_strategies:
        container_children = [
            html.Div(
                "No active strategies.",
                style={"fontSize": "0.8rem", "color": "#888888"},
            )
        ]
    else:
        container_children = [
            _build_active_row(
                uid=s["uid"],
                name=s["name"],
                is_selected=s["is_selected"],
                is_saved=s["is_saved"],
                portfolio_count=s["portfolio_count"],
            )
            for s in active_strategies
        ]

    count_badge = f" ({len(active_strategies)})"

    # Keep store in sync with the current snapshot
    active_store = active_strategies

    return container_children, count_badge, active_store, new_values



@callback(
    Output("p1-universe-list-container", "children"),
    Output("p1-universe-count-badge", "children"),
    Output("p1-universe-list-store", "data"),
    Input("p1-strategy-checklist", "options"),
    Input("p1-universe-search", "value"),
    Input({"type": "universe-row-checkbox", "uid": ALL}, "value"),
    Input("p1-universe-select-all", "value"),
    State("p1-universe-list-store", "data"),
)
def update_universe_list_display(
    checklist_options,
    search_value,
    checkbox_values,
    select_all_value,
    universe_store,
):
    """
    Render the Universe List based on saved strategies in registry.

    Universe List shows all strategies with is_saved=True.
    Supports:
      - Live search filtering
      - Multi-select via row checkboxes
      - 'Select all (filtered)' toggle
    """
    triggered_id = ctx.triggered_id

    # Normalise inputs
    universe_store = universe_store or []
    search_value = (search_value or "").lower().strip()
    checklist_options = checklist_options or []
    checkbox_values = checkbox_values or []

    # --- Determine which UIDs are currently active (in Active list) ---
    active_uids = set()
    for opt in checklist_options:
        sid = opt.get("value")
        meta = p1_strategy_store.get(sid, {})
        uid = meta.get("uid") or derive_uid_from_filepath(sid)
        active_uids.add(uid)

    # --- Build base universe list from registry (saved strategies only) ---
    registry = load_registry()
    all_strategies = registry.get("strategies", [])

    # Map stored multi-select flags by uid
    stored_selected = {
        item.get("uid"): item.get("is_multi_selected", False)
        for item in universe_store
        if item.get("uid")
    }

    universe_strategies = []
    for s in all_strategies:
        if not s.get("is_saved", False):
            continue

        uid = s.get("uid")
        if not uid:
            continue

        name = s.get("name", uid)

        # Apply search filter
        if search_value and search_value not in name.lower():
            continue

        # Check if active
        is_active = uid in active_uids

        # Portfolio info
        portfolios = get_portfolios_for_uid(uid, registry)
        portfolio_count = len(portfolios)
        portfolio_names = [p.get("name", p.get("id", "")) for p in portfolios]

        # Default multi-select state from store
        is_multi_selected = stored_selected.get(uid, False)

        universe_strategies.append(
            {
                "uid": uid,
                "name": name,
                "is_active": is_active,
                "is_multi_selected": is_multi_selected,
                "portfolio_count": portfolio_count,
                "portfolio_names": portfolio_names,
            }
        )

    # Sort by name for stable ordering
    universe_strategies.sort(key=lambda x: x["name"].lower())

    # --- Apply user interaction: row checkbox or Select-all ---
    if isinstance(triggered_id, dict) and triggered_id.get("type") == "universe-row-checkbox":
        # A single row checkbox was toggled; use checkbox_values (aligned by order)
        for idx, row in enumerate(universe_strategies):
            if idx < len(checkbox_values):
                # If row is active, keep it unselected & disabled; otherwise follow checkbox
                row["is_multi_selected"] = bool(checkbox_values[idx]) if not row["is_active"] else False

    elif triggered_id == "p1-universe-select-all":
        # Select/Deselect all *visible* rows (respecting active flag)
        select_all_flag = bool(select_all_value)
        for row in universe_strategies:
            row["is_multi_selected"] = select_all_flag if not row["is_active"] else False

    else:
        # Triggered by search or changes in active list; keep stored selection
        for row in universe_strategies:
            row["is_multi_selected"] = stored_selected.get(row["uid"], False)

    # --- Build container children ---
    if not universe_strategies:
        if search_value:
            container_children = [
                html.Div(
                    f"No strategies matching '{search_value}'.",
                    style={"fontSize": "0.8rem", "color": "#888888"},
                )
            ]
        else:
            container_children = [
                html.Div(
                    "No strategies in universe.",
                    style={"fontSize": "0.8rem", "color": "#888888"},
                )
            ]
    else:
        container_children = [
            _build_universe_row(
                uid=s["uid"],
                name=s["name"],
                is_multi_selected=s["is_multi_selected"],
                is_active=s["is_active"],
                portfolio_count=s["portfolio_count"],
                portfolio_names=s["portfolio_names"],
            )
            for s in universe_strategies
        ]

    count_badge = f" ({len(universe_strategies)})"

    # Write back the current visible universe snapshot to the store
    new_universe_store = [
        {
            "uid": s["uid"],
            "name": s["name"],
            "is_active": s["is_active"],
            "is_multi_selected": s["is_multi_selected"],
            "portfolio_count": s["portfolio_count"],
            "portfolio_names": s["portfolio_names"],
        }
        for s in universe_strategies
    ]

    return container_children, count_badge, new_universe_store



@callback(
    Output("p1-universe-select-all", "value"),
    Input("p1-universe-select-all", "value"),
    State("p1-universe-list-store", "data"),
    prevent_initial_call=True,
)
def toggle_universe_select_all(select_all_value, universe_store):
    """
    Handle Select/Deselect All toggle for Universe list.
    This updates the is_multi_selected state for all visible (filtered) strategies.
    """
    # The actual selection update happens through the checkbox values
    # This callback just maintains the checkbox state
    return select_all_value


@callback(
    Output("p1-strategy-checklist", "options", allow_duplicate=True),
    Output("p1-strategy-checklist", "value", allow_duplicate=True),
    Output("app-notifications", "data", allow_duplicate=True),
    Input("p1-universe-move-to-active-btn", "n_clicks"),
    Input({"type": "universe-row-quickadd", "uid": ALL}, "n_clicks"),
    State("p1-universe-list-store", "data"),
    State("p1-strategy-checklist", "options"),
    State("p1-strategy-checklist", "value"),
    State("app-notifications", "data"),
    prevent_initial_call=True,
)
def move_to_active_from_universe(
    move_btn_clicks,
    quickadd_clicks,
    universe_store,
    existing_options,
    existing_values,
    existing_notifications,
):
    """
    Move strategies from Universe to Active list.
    
    - Bulk move: Move all selected (is_multi_selected) strategies.
    - Quick-add: Move a single strategy via the + button.
    
    Sets is_active=True, is_selected=False for moved strategies.
    """
    triggered_id = ctx.triggered_id
    
    existing_options = existing_options or []
    existing_values = existing_values or []
    existing_notifications = existing_notifications or []
    universe_store = universe_store or []
    
    # Get currently active UIDs
    active_uids = set()
    for opt in existing_options:
        sid = opt.get("value")
        meta = p1_strategy_store.get(sid, {})
        uid = meta.get("uid") or derive_uid_from_filepath(sid)
        active_uids.add(uid)
    
    uids_to_add = []
    skip_notifications = []
    
    if triggered_id == "p1-universe-move-to-active-btn":
        # Bulk move selected strategies
        for item in universe_store:
            if item.get("is_multi_selected") and not item.get("is_active"):
                uids_to_add.append(item.get("uid"))
            elif item.get("is_multi_selected") and item.get("is_active"):
                skip_notifications.append(
                    create_notification("SKIP", f"[SKIP] '{item.get('name', item.get('uid'))}' already active.")
                )
    
    elif isinstance(triggered_id, dict) and triggered_id.get("type") == "universe-row-quickadd":
        # Quick-add single strategy
        uid = triggered_id.get("uid")
        if uid and uid not in active_uids:
            uids_to_add.append(uid)
        elif uid:
            skip_notifications.append(
                create_notification("SKIP", f"[SKIP] Strategy already active.")
            )
    
    if not uids_to_add:
        updated_notifications = existing_notifications + skip_notifications
        return no_update, no_update, updated_notifications
    
    # Add strategies to the checklist (making them active)
    registry = load_registry()
    new_options = list(existing_options)
    option_values = {opt["value"] for opt in new_options}
    
    for uid in uids_to_add:
        # Get strategy info from registry
        strategy = get_strategy_by_uid(uid, registry)
        if not strategy:
            continue
        
        # Use internal path as the id for consistency
        internal_path = str(get_internal_strategy_path(uid))
        
        if internal_path in option_values:
            continue  # Already in options
        
        name = strategy.get("name", uid)
        
        # Load strategy data into memory store
        try:
            df = pd.read_csv(internal_path)
            cache_cols = ["Date Closed", "P/L"]
            for extra_col in ["Margin Req.", "Date Opened", "Time Opened", "Premium", "Gap", "Movement"]:
                if extra_col in df.columns:
                    cache_cols.append(extra_col)
            
            strategy_meta = dict(strategy)
            strategy_meta["df"] = df[cache_cols].copy()
            strategy_meta["file_path"] = internal_path
            strategy_meta["is_active"] = True
            strategy_meta["is_selected"] = False
            
            p1_strategy_store[internal_path] = strategy_meta
            p1_strategy_store[uid] = strategy_meta
        except (FileNotFoundError, PermissionError, pd.errors.EmptyDataError, pd.errors.ParserError):
            # Skip if file doesn't exist, is inaccessible, or can't be parsed
            continue
        
        new_options.append({"label": name, "value": internal_path})
    
    # Values remain unchanged (new strategies are added with is_selected=False)
    updated_notifications = existing_notifications + skip_notifications
    
    return new_options, existing_values, updated_notifications


@callback(
    Output("p1-strategy-checklist", "options", allow_duplicate=True),
    Output("p1-strategy-checklist", "value", allow_duplicate=True),
    Input({"type": "active-row-remove", "uid": ALL}, "n_clicks"),
    State("p1-strategy-checklist", "options"),
    State("p1-strategy-checklist", "value"),
    State("p1-active-list-store", "data"),
    prevent_initial_call=True,
)
def remove_from_active_list(
    remove_clicks,
    existing_options,
    existing_values,
    active_store,
):
    """
    Remove a strategy from the Active list when × button is clicked.
    
    - Sets is_active=False, is_selected=False.
    - If is_saved=True: remains in Universe.
    - If is_saved=False: removed from memory entirely.
    """
    triggered_id = ctx.triggered_id
    
    if not isinstance(triggered_id, dict) or triggered_id.get("type") != "active-row-remove":
        return no_update, no_update
    
    # Check if any button was actually clicked
    if not any(n for n in (remove_clicks or []) if n):
        return no_update, no_update
    
    uid_to_remove = triggered_id.get("uid")
    if not uid_to_remove:
        return no_update, no_update
    
    existing_options = existing_options or []
    existing_values = existing_values or []
    active_store = active_store or []
    
    # Find the strategy's sid (checklist value)
    sid_to_remove = None
    is_saved = False
    
    for item in active_store:
        if item.get("uid") == uid_to_remove:
            sid_to_remove = item.get("sid")
            is_saved = item.get("is_saved", False)
            break
    
    if not sid_to_remove:
        # Try to find by uid in options
        for opt in existing_options:
            meta = p1_strategy_store.get(opt["value"], {})
            if meta.get("uid") == uid_to_remove:
                sid_to_remove = opt["value"]
                is_saved = meta.get("is_saved", False)
                break
    
    if not sid_to_remove:
        return no_update, no_update
    
    # Remove from options
    new_options = [opt for opt in existing_options if opt["value"] != sid_to_remove]
    new_values = [v for v in existing_values if v != sid_to_remove]
    
    # If not saved, remove from memory store
    if not is_saved:
        p1_strategy_store.pop(sid_to_remove, None)
        p1_strategy_store.pop(uid_to_remove, None)
    else:
        # Update flags in store
        if sid_to_remove in p1_strategy_store:
            p1_strategy_store[sid_to_remove]["is_active"] = False
            p1_strategy_store[sid_to_remove]["is_selected"] = False
    
    return new_options, new_values


# @callback(
#     Output("p1-strategy-checklist", "value", allow_duplicate=True),
#     Input({"type": "active-row-checkbox", "uid": ALL}, "value"),
#     State("p1-strategy-checklist", "options"),
#     State("p1-strategy-checklist", "value"),
#     State("p1-active-list-store", "data"),
#     prevent_initial_call=True,
# )
# def sync_active_checkbox_to_checklist(
#     checkbox_values,
#     existing_options,
#     existing_values,
#     active_store,
# ):
#     """
#     Sync the Active row checkboxes to the main checklist values.
#     This ensures is_selected state is properly tracked.
#     """
#     triggered_id = ctx.triggered_id
    
#     if not isinstance(triggered_id, dict) or triggered_id.get("type") != "active-row-checkbox":
#         return no_update
    
#     uid = triggered_id.get("uid")
#     if not uid:
#         return no_update
    
#     existing_options = existing_options or []
#     existing_values = existing_values or []
#     active_store = active_store or []
    
#     # Find the sid for this uid
#     sid = None
#     for item in active_store:
#         if item.get("uid") == uid:
#             sid = item.get("sid")
#             break
    
#     if not sid:
#         return no_update
    
#     # Find the checkbox value that was just changed
#     new_is_selected = None
#     for opt in existing_options:
#         meta = p1_strategy_store.get(opt["value"], {})
#         if meta.get("uid") == uid or opt["value"] == sid:
#             # Get the index in active_store to find checkbox value
#             for i, item in enumerate(active_store):
#                 if item.get("uid") == uid:
#                     if i < len(checkbox_values):
#                         new_is_selected = checkbox_values[i]
#                     break
#             break
    
#     if new_is_selected is None:
#         return no_update
    
#     # Update values
#     new_values = list(existing_values)
#     if new_is_selected and sid not in new_values:
#         new_values.append(sid)
#     elif not new_is_selected and sid in new_values:
#         new_values.remove(sid)
    
#     return new_values
