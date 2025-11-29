# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:09:09 2025

@author: mauro
"""

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output

from core.sh_layout import (
    build_data_input_section,
    build_strategy_sidebar,
    ROOT_DATA_DIR,
    _list_immediate_subfolders,
)
from core.registry import list_portfolios

from pages.page1 import build_phase1_right_panel
from pages.page2 import build_phase2_right_panel




PORT = 8050



# -----------------------------------------------------------------------------
# Create Dash app
# -----------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],  # dark theme
    suppress_callback_exceptions=True,         # allow per-page layouts
)

app.title = "Portfolio26"


# -----------------------------------------------------------------------------
# Page layout functions (for now, simple placeholders)
# Later you can move each of these into a separate module under /pages.
# -----------------------------------------------------------------------------


def layout_page_3_portfolio_compare():
    return dbc.Container(
        [
            html.H3("Phase 2 – Portfolio Comparisons", className="mb-3"),
            html.P("TODO: compare multiple candidate portfolios side by side."),
        ],
        fluid=True,
    )


def layout_page_4_ml_utils():
    return dbc.Container(
        [
            html.H3("Phase 2–3 – ML Utilities", className="mb-3"),
            html.P("TODO: file conversions, dataset preparation, consistency checks."),
        ],
        fluid=True,
    )


def layout_page_5_ml_output():
    return dbc.Container(
        [
            html.H3("Phase 3 – ML Output Analysis", className="mb-3"),
            html.P("TODO: ML vs baseline portfolio metrics and charts."),
        ],
        fluid=True,
    )


# -----------------------------------------------------------------------------
# Main layout: navbar + tabs + shared loader/sidebar + right panel
# -----------------------------------------------------------------------------
# Build folder list and checklist for shared loader
subfolders = _list_immediate_subfolders(ROOT_DATA_DIR)

if not subfolders:
    folder_info = html.Div(
        [
            html.Div(
                "No subfolders found under root data folder:",
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

# Portfolio dropdown options (same registry as Phase 1)
portfolios = list_portfolios()
portfolio_options = [
    {
        "label": p.get("name", p.get("id", "Unnamed portfolio")),
        "value": p.get("id"),
    }
    for p in portfolios
    if p.get("id") is not None
]

app.layout = dbc.Container(
    [
        dbc.Navbar(
            dbc.Container(
                [
                    dbc.NavbarBrand("Portfolio26 – Research & ML", className="ms-2"),
                ],
                fluid=True,
            ),
            color="dark",
            dark=True,
            className="mb-3",
        ),

        # Main top-level tabs
        dbc.Tabs(
            id="main-tabs",
            active_tab="tab-1",
            children=[
                dbc.Tab(label="Strategies R&D", tab_id="tab-1"),
                dbc.Tab(label="Portfolio Analytics", tab_id="tab-2"),
                dbc.Tab(label="Portfolio Comparisons", tab_id="tab-3"),
                dbc.Tab(label="ML Utilities", tab_id="tab-4"),
                dbc.Tab(label="ML Output Analysis", tab_id="tab-5"),
            ],
        ),

        # Shared state for current portfolio (used by shared callbacks)
        dcc.Store(id="p1-current-portfolio-id", data=None),

        # Shared loader / data input section (collapsible)
        build_data_input_section(folder_info, folder_checklist, portfolio_options),

        dbc.Row(
            [
                # Shared strategy list sidebar (always present)
                build_strategy_sidebar(),

                # Right-hand panel: content depends on active main tab
                dbc.Col(
                    html.Div(id="tab-content", className="mt-3"),
                    width=9,
                ),
            ]
        ),
    ],
    fluid=True,
)


# -----------------------------------------------------------------------------
# Callbacks: switch tab content
# -----------------------------------------------------------------------------
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
)
def render_tab_content(active_tab):
    if active_tab == "tab-1":
        # Phase 1: use right-hand analytics panel only
        return build_phase1_right_panel()
    elif active_tab == "tab-2":
        # Phase 2: portfolio-level right-hand panel
        return build_phase2_right_panel()
    elif active_tab == "tab-3":
        # For now, keep existing placeholder as the right panel
        return layout_page_3_portfolio_compare()
    elif active_tab == "tab-4":
        return layout_page_4_ml_utils()
    elif active_tab == "tab-5":
        return layout_page_5_ml_output()
    # fallback
    return html.Div("Unknown tab")



# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=PORT, host="127.0.0.1")
