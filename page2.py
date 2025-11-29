# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 10:18:10 2025

@author: mauro
"""

# -*- coding: utf-8 -*-
"""
Phase 2 – Portfolio Construction

Main page layout for Phase 2, reusing shared loader + strategy list.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc

from core.sh_layout import (
    build_data_input_section,
    build_strategy_sidebar,
    ROOT_DATA_DIR,
    _list_immediate_subfolders,
)
from core.registry import list_portfolios


def build_phase2_right_panel():
    """
    Phase 2 right-hand panel (Portfolio Analytics / Allocation Scenarios / Optimizer).
    Does NOT include loader or strategy list. Reused later by main.py.
    """
    return dbc.Card(
        [
            dbc.CardHeader("Portfolio-level analysis and optimizer"),
            dbc.CardBody(
                [
                    dcc.Tabs(
                        id="p2-analysis-tabs",
                        value="p2-analytics",
                        style={"backgroundColor": "#222222"},
                        children=[
                            dcc.Tab(
                                label="Portfolio Analytics",
                                value="p2-analytics",
                                children=html.Div(
                                    "Phase 2 – Portfolio Analytics (placeholder)",
                                    style={"padding": "1rem", "fontSize": "0.9rem"},
                                ),
                                style={
                                    "backgroundColor": "#333333",
                                    "color": "#BBBBBB",
                                    "padding": "8px 12px",
                                },
                                selected_style={
                                    "backgroundColor": "#555555",
                                    "color": "#FFFFFF",
                                    "padding": "8px 12px",
                                },
                            ),
                            dcc.Tab(
                                label="Allocation Scenarios",
                                value="p2-scenarios",
                                children=html.Div(
                                    "Phase 2 – Allocation Scenarios (placeholder)",
                                    style={"padding": "1rem", "fontSize": "0.9rem"},
                                ),
                                style={
                                    "backgroundColor": "#333333",
                                    "color": "#BBBBBB",
                                    "padding": "8px 12px",
                                },
                                selected_style={
                                    "backgroundColor": "#555555",
                                    "color": "#FFFFFF",
                                    "padding": "8px 12px",
                                },
                            ),
                            dcc.Tab(
                                label="Portfolio Optimizer",
                                value="p2-optimizer",
                                children=html.Div(
                                    "Phase 2 – Portfolio Optimizer (placeholder)",
                                    style={"padding": "1rem", "fontSize": "0.9rem"},
                                ),
                                style={
                                    "backgroundColor": "#333333",
                                    "color": "#BBBBBB",
                                    "padding": "8px 12px",
                                },
                                selected_style={
                                    "backgroundColor": "#555555",
                                    "color": "#FFFFFF",
                                    "padding": "8px 12px",
                                },
                            ),
                        ],
                    )
                ]
            ),
        ]
    )


def layout_page_2_portfolio():
    """
    Layout for Phase 2 – Portfolio Construction.

    Left/top:
        - Shared 'Data Input – Strategies' collapsible panel
        - Shared 'Strategy List' sidebar

    Right:
        - Phase 2-specific tabs (Portfolio Analytics, placeholder, Optimizer)
    """

    # ------------------------------------------------------------------
    # Folder list + checklist (same idea as Phase 1)
    # ------------------------------------------------------------------
    subfolders = _list_immediate_subfolders(ROOT_DATA_DIR)

    if not subfolders:
        folder_info = html.Div(
            [
                html.Div(
                    f"No subfolders found under root data folder:",
                    style={"fontSize": "0.85rem"},
                ),
                html.Code(ROOT_DATA_DIR, style={"fontSize": "0.8rem"}),
            ],
            style={"marginBottom": "0.5rem"},
        )
        folder_checklist = html.Div(
            "No folders available for scanning.",
            style={"fontSize": "0.8rem", "color": "#AAAAAA"},
        )
    else:
        folder_info = html.Div(
            [
                html.Div(
                    "Root data folder for strategies:",
                    style={"fontSize": "0.85rem"},
                ),
                html.Code(ROOT_DATA_DIR, style={"fontSize": "0.8rem"}),
                html.Div(
                    "Select one or more subfolders to scan for CSV strategy files.",
                    style={"fontSize": "0.8rem", "marginTop": "0.25rem"},
                ),
            ],
            style={"marginBottom": "0.5rem"},
        )

        # Checklist options: show only folder name in label, full path as value
        folder_checklist = dcc.Checklist(
            id="p1-folder-checklist",
            options=[
                {"label": name, "value": full_path} for name, full_path in subfolders
            ],
            value=[],
            labelStyle={"display": "block"},
            style={"fontSize": "0.8rem", "maxHeight": "200px", "overflowY": "auto"},
        )

    # ------------------------------------------------------------------
    # Portfolio dropdown options (same registry as Phase 1)
    # ------------------------------------------------------------------
    portfolios = list_portfolios()
    portfolio_options = [
        {
            "label": p.get("name", p.get("id", "Unnamed portfolio")),
            "value": p.get("id"),
        }
        for p in portfolios
        if p.get("id") is not None
    ]

    # ------------------------------------------------------------------
    # Assemble layout
    # ------------------------------------------------------------------
    return dbc.Container(
        [
            html.H3("Phase 2 – Portfolio Construction", className="mb-3"),

            # We still rely on shared p1-* IDs for loader + strategy list,
            # so no new Store is needed here for selection; Phase 2 analytics
            # will read from the same components as Phase 1.
            # If we need Phase-2-specific state later, we can add p2-* stores.

            # Shared portfolio identity store (needed by shared callbacks)
            dcc.Store(id="p1-current-portfolio-id", data=None),
            
            # Shared Data Input section (collapsible)
            build_data_input_section(folder_info, folder_checklist, portfolio_options),

            html.Hr(),

            dbc.Row(
                [
                    # LEFT: shared Strategy List sidebar
                    build_strategy_sidebar(),

                    # RIGHT: Phase 2 tabs
                    dbc.Col(
                        build_phase2_right_panel(),
                        width=9,
                    ),

                ]
            ),
        ],
        fluid=True,
    )
