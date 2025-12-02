# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 06:27:27 2025

@author: mauro
"""

# core/sh_layout.py

import os
import io
import base64
import pandas as pd
from datetime import datetime
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, ctx

from core.registry import (
    get_registry,
    add_or_update_strategy,
    add_or_update_portfolio,
    list_portfolio_options,
    list_portfolios,
    get_strategy,
    get_portfolio,
    set_phase1_active_flags,
    get_phase1_active_uids,
    list_strategies,
)


# In-memory store for strategies loaded in this session (shared across pages)
# Key: strategy_id, Value: dict with metadata (id, name, file_path, cached df, etc.)
p1_strategy_store = {}

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
    """
    return dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader(
                    "Strategy List (Portfolio: Unsaved Portfolio)",
                    id="p1-strategy-list-header",
                ),
                dbc.CardBody(
                    [
                        # Hidden store just to satisfy Dash output requirement when updating registry
                        dcc.Store(id="p1-strategy-registry-sync"),
                        dbc.Checkbox(
                            id="p1-strategy-select-all",
                            value=True,
                            label="Select / deselect all",
                            style={
                                "fontSize": "0.8rem",
                                "marginBottom": "0.4rem",
                            },
                        ),
                        dcc.Checklist(
                            id="p1-strategy-checklist",
                            options=[],
                            value=[],
                            labelStyle={
                                "display": "block",
                                "marginBottom": "0.15rem",
                            },
                            style={
                                "maxHeight": "400px",
                                "overflowY": "auto",
                                "fontSize": "0.85rem",
                            },
                        ),
                        html.Div(
                            id="p1-strategy-summary",
                            children="No strategies loaded yet.",
                            style={
                                "fontSize": "0.8rem",
                                "marginTop": "0.4rem",
                                "color": "#AAAAAA",
                            },
                        ),
                    ]
                ),
            ],
            style={"position": "sticky", "top": "80px"},
        ),
        width=3,
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
    registry = get_registry()
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
        registry = get_registry() or {}
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
    existing_strats = list_strategies()
    strat_map = {s.get("id"): s for s in existing_strats if "id" in s}


        # ---------- Ensure each selected strategy is present in registry ----------
    for sid in selected_strategy_ids:
        # Start from whatever we have in the in-memory store
        meta = (p1_strategy_store.get(sid) or {}).copy()

        # Always enforce the id
        meta["id"] = sid

        # file_path: if missing, default to the id (for folder-based strategies)
        if not meta.get("file_path"):
            meta["file_path"] = sid

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

        # Keep store and registry in sync
        p1_strategy_store[sid] = meta
        add_or_update_strategy(meta)


    # ---------- Add or update portfolio in registry ----------
    portfolio_obj = {
        "id": portfolio_id,
        "name": portfolio_name_final,
        "strategy_ids": selected_strategy_ids,
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
        f"{len(selected_strategy_ids)} strategies."
    )

    # current_portfolio_id becomes portfolio_id after any successful save
    return status, portfolio_id, options, value


@callback(
    Output("p1-strategy-list-header", "children"),
    Input("p1-current-portfolio-id", "data"),
)
def update_strategy_list_header(current_portfolio_id):
    """
    Show which portfolio is currently active in the Strategy List header.
    """
    if not current_portfolio_id:
        return "Strategy List (Portfolio: Unsaved Portfolio)"

    portfolio = get_portfolio(current_portfolio_id)
    if portfolio:
        name = portfolio.get("name", current_portfolio_id)
    else:
        name = current_portfolio_id

    return f"Strategy List (Portfolio: {name})"

@callback(
    Output("p1-strategy-checklist", "options"),
    Output("p1-strategy-checklist", "value"),
    Output("p1-strategy-select-all", "value"),
    Output("p1-strategy-summary", "children"),
    Output("p1-current-portfolio-id", "data"),
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
):
    """
    Dispatcher for:
      - Loading strategies from CSVs (Load Selected Strategies)
      - Loading an existing portfolio
      - Selecting/deselecting all

    IMPORTANT:
    - This callback never clears the registry.
    - CSV loads merge into the current list; they do not replace it.
    """
    triggered_id = ctx.triggered_id

    # Initial state: nothing has happened yet
    if triggered_id is None:
        return [], [], False, "No strategies loaded yet.", current_portfolio_id


    existing_options = existing_options or []
    existing_values = existing_values or []

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
        
                # Use the stored path as the strategy id (same convention as folder-based)
                strategy_id = stored_path
                # Decide which columns to cache: always Date Closed + P/L, plus Margin Req. if present
                cache_cols = ["Date Closed", "P/L"]
                if "Margin Req." in df.columns:
                    cache_cols.append("Margin Req.")
                
                strategy_meta = {
                    "id": strategy_id,
                    "name": f"SINGLE. {os.path.splitext(fname)[0]}",
                    "file_path": stored_path,
                    "folder": "dataupl",
                    "source": "upload",
                    "n_rows": n_rows,
                    # Cache a trimmed DataFrame for analytics (including Margin Req. when available)
                    "df": df[cache_cols].copy(),
                }

        
                p1_strategy_store[strategy_id] = strategy_meta
        
                if strategy_id not in option_map:
                    option_map[strategy_id] = (
                        strategy_id,
                        strategy_meta["name"],
                    )
                    new_ids.append(strategy_id)


        # If we still have nothing, report and keep context
        if not option_map:
            summary = "No strategies loaded (no valid CSVs found)."
            if errors:
                summary += " " + " ".join(errors)
            return [], [], False, summary, current_portfolio_id

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
                f"Added {len(new_ids)} new strategies from folders/uploads. "
                f"{len(selected_values)} of {len(options)} strategies selected for Phase 1."
            )
        else:
            base_summary = (
                f"No new strategies added. "
                f"{len(selected_values)} of {len(options)} strategies selected for Phase 1."
            )

        if errors:
            base_summary += " " + " ".join(errors)

        # IMPORTANT: keep current_portfolio_id unchanged
        return options, selected_values, select_all_flag, base_summary, current_portfolio_id

    # ----------------------------------------------------------
    # CASE 2: Load existing portfolio (from registry)
    # ----------------------------------------------------------
    if triggered_id == "p1-load-portfolio-btn":
        options = existing_options or []

        if not selected_portfolio_id:
            summary = "No portfolio selected."
            current_values = []
            return options, current_values, False, summary, None

        portfolio = get_portfolio(selected_portfolio_id)
        if not portfolio:
            summary = f"Portfolio '{selected_portfolio_id}' not found in registry."
            current_values = []
            return options, current_values, False, summary, None

        # All strategies currently known in the registry
        all_strats = list_strategies()
        if not all_strats:
            summary = (
                "Registry has no strategies. "
                "Save at least one portfolio before using this feature."
            )
            return [], [], False, summary, selected_portfolio_id

        portfolio_ids = set(portfolio.get("strategy_ids", []))

        # Build checklist options from all strategies in registry
        options = [
            {
                "label": s.get("name", s.get("id", "UNKNOWN")),
                "value": s["id"],
            }
            for s in sorted(
                all_strats, key=lambda s: s.get("name", s.get("id", ""))
            )
        ]

        selected_values = [
            s["id"] for s in all_strats if s["id"] in portfolio_ids
        ]

        n_missing = len(portfolio_ids) - len(selected_values)

        select_all_flag = len(selected_values) == len(options)

        summary = (
            f"Loaded portfolio '{portfolio.get('name', selected_portfolio_id)}' – "
            f"{len(selected_values)} strategies selected."
        )
        if n_missing > 0:
            summary += f" ({n_missing} strategy id(s) in portfolio not found in registry.)"

        # Portfolio context is set to the loaded portfolio id
        return options, selected_values, select_all_flag, summary, selected_portfolio_id

    # ----------------------------------------------------------
    # CASE 3: Select-all checkbox toggled
    # ----------------------------------------------------------
    if triggered_id == "p1-strategy-select-all":
        options = existing_options or []
        if not options:
            return [], [], False, "No strategies loaded yet.", current_portfolio_id

        if select_all_value:
            values = [opt["value"] for opt in options]
        else:
            values = []

        summary = f"{len(values)} of {len(options)} strategies selected for Phase 1."
        return options, values, select_all_value, summary, current_portfolio_id

    # Fallback (shouldn't hit here)
    options = existing_options or []
    if not options:
        return [], [], False, "No strategies loaded yet.", current_portfolio_id
    return options, [], False, "No strategies loaded yet.", current_portfolio_id
