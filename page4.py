# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 13:17:54 2025

@author: mauro
"""

# -*- coding: utf-8 -*-
"""
Phase 3 – ML Utilities (Page 4)

Right-panel layout + minimal wiring for ML data build.
The actual data build logic will live outside /pages (e.g., portfolio26/ml/).
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State

from ml.data_prep import build_ml_panel_dataset



def build_phase3_right_panel():
    """
    Returns the right-side panel content for Tab-4 (ML Utilities).

    This follows the same architecture as:
      - pages.page1.build_phase1_right_panel
      - pages.page2.build_phase2_right_panel
    """
    return html.Div(
        [
            # Header row
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H4("ML Utilities", className="mb-1"),
                                html.Div(
                                    "Dataset builder and ML run controls (Phase 3).",
                                    style={"fontSize": "0.85rem", "color": "#A8A8A8"},
                                ),
                            ]
                        ),
                        width=9,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Span("Phase 3 score: ", style={"fontSize": "0.85rem"}),
                                html.B("N/A", id="p3-score-badge"),
                            ],
                            style={"textAlign": "right", "paddingTop": "6px"},
                        ),
                        width=3,
                    ),
                ],
                className="mb-3",
            ),

            # -----------------------------------------------------------------
            # ML Data Builder - OLD CPO - Strategy level
            # -----------------------------------------------------------------
            dbc.Card(
                [
                    dbc.CardHeader(html.B("ML Data Builder (Old CPO - strategy level)")),
                    dbc.CardBody(
                        [
                            html.Div(
                                "Build the consolidated ML panel dataset for the currently selected portfolio.",
                                style={"fontSize": "0.85rem", "color": "#B0B0B0"},
                                className="mb-2",
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Checklist(
                                            id="ml-save-artifacts-toggle",
                                            options=[
                                                {
                                                    "label": "Save dataset artifact to disk (CSV)",
                                                    "value": "save",
                                                }
                                            ],
                                            value=["save"],  # default ON for development
                                            switch=True,
                                        ),
                                        width=7,
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Build ML Snapshot Dataset",
                                            id="ml-build-panel-btn",
                                            color="primary",
                                            n_clicks=0,
                                            className="w-100",
                                        ),
                                        width=5,
                                    ),
                                ],
                                className="mb-2",
                            ),

                            dbc.Alert(
                                id="ml-build-status",
                                children="No dataset built yet.",
                                color="secondary",
                                dismissable=False,
                                is_open=True,
                                style={"fontSize": "0.85rem"},
                            ),

                            # Placeholders for future: preview / stats
                            html.Div(
                                id="ml-build-summary",
                                children="",
                                style={"fontSize": "0.85rem", "color": "#B0B0B0"},
                                className="mt-2",
                            ),

                            # Store to persist latest built dataset path / metadata (future use)
                            dcc.Store(id="ml-last-built-dataset-meta", data=None),
                        ]
                    ),
                ],
                className="mb-3",
            ),

            # -----------------------------------------------------------------
            # Future sections placeholders (no logic yet)
            # -----------------------------------------------------------------
            dbc.Card(
                [
                    dbc.CardHeader(html.B("FWA Runner (placeholder)")),
                    dbc.CardBody(
                        html.Div(
                            "This section will host the walk-forward run controls and outputs.",
                            style={"fontSize": "0.85rem", "color": "#B0B0B0"},
                        )
                    ),
                ]
            ),
        ],
        className="mt-3",
    )


@callback(
    Output("ml-build-status", "children"),
    Output("ml-build-status", "color"),
    Output("ml-build-summary", "children"),
    Output("ml-last-built-dataset-meta", "data"),
    Input("ml-build-panel-btn", "n_clicks"),
    State("ml-save-artifacts-toggle", "value"),
    State("p1-strategy-checklist", "value"),
    State("p2-weights-store", "data"),
    State("p1-current-portfolio-id", "data"),
    prevent_initial_call=True,
)
def ml_build_panel_dataset(n_clicks, save_toggle, selected_strategy_ids, weights_store, portfolio_id):

    """
    Minimal wiring callback: confirms UI integration.
    Actual dataset build logic will be implemented later in portfolio26/ml/.
    """
    save_to_disk = bool(save_toggle) and ("save" in (save_toggle or []))

    selected_strategy_ids = selected_strategy_ids or []
    if not selected_strategy_ids:
        return (
            "ERROR: No strategies selected. Select strategies in the Active list first, then retry.",
            "danger",
            "",
            None,
        )
    
    try:
        result = build_ml_panel_dataset(
            selected_strategy_ids=selected_strategy_ids,
            weights_store=weights_store or {},
            source_portfolio_id=portfolio_id,
            source_portfolio_name=None,  # optional for later if you store portfolio name somewhere
            save_to_disk=save_to_disk,
        )
    except Exception as e:
        return (
            f"ERROR: dataset build failed: {e}",
            "danger",
            "",
            None,
        )
    
    # If builder produced no rows, show warnings
    if not result.get("dataset_path") and int(result.get("n_rows", 0)) == 0:
        warn_lines = result.get("errors", [])
        warn_txt = "\n".join(warn_lines[:8])  # keep UI compact
        return (
            "ERROR: No dataset created (no usable rows).",
            "danger",
            warn_txt,
            result,
        )
    
    # Success
    snap = result.get("snapshot_id")
    path = result.get("dataset_path")
    n_rows = result.get("n_rows")
    n_strat = len(result.get("selected_uids", []))
    dt_min = result.get("min_open_dt")
    dt_max = result.get("max_open_dt")
    
    msg = f"OK: Built ML panel dataset. snapshot_id={snap}"
    summary = (
        f"Rows={n_rows}, Strategies={n_strat}, Range={dt_min} → {dt_max}\n"
        f"Saved={save_to_disk}, Path={path}"
    )
    
    # Show warnings (if any) but do not fail
    errs = result.get("errors") or []
    if errs:
        summary = summary + "\nWarnings:\n" + "\n".join(errs[:8])
    
    return msg, "success", summary, result

