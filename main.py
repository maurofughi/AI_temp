# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:09:09 2025

@author: mauro
"""

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, ctx
from dash.dependencies import Input, Output, State, ALL

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


# -----------------------------------------------------------------------------
# Notification System Components
# -----------------------------------------------------------------------------
def build_notification_bell():
    """
    Build the notification bell icon with badge for the header.
    """
    return html.Div(
        [
            html.Div(
                [
                    # Bell icon (using Unicode bell character)
                    html.Span("🔔", className="notification-bell-icon"),
                    # Badge (count)
                    dbc.Badge(
                        id="notification-badge",
                        children="0",
                        className="notification-badge notification-badge-skip",
                        style={"display": "none"},
                    ),
                ],
                id="notification-bell-btn",
                className="notification-bell-container",
                title="Notifications",
            ),
        ],
        style={"marginLeft": "auto"},
    )


def build_notification_panel():
    """
    Build the notification panel (Offcanvas) that slides in from the right.
    """
    return dbc.Offcanvas(
        id="notification-panel",
        title="Notifications",
        is_open=False,
        placement="end",  # slide in from right
        className="notification-panel",
        style={"width": "400px"},
        children=[
            # Notification list container
            html.Div(
                id="notification-list",
                className="notification-list",
                children=[
                    html.Div(
                        "No notifications",
                        className="notification-empty",
                    )
                ],
            ),
            # Footer with Dismiss All button
            html.Div(
                dbc.Button(
                    "Dismiss All",
                    id="notification-dismiss-all-btn",
                    color="secondary",
                    size="sm",
                    outline=True,
                    style={"width": "100%"},
                ),
                id="notification-panel-footer",
                className="notification-panel-footer",
                style={"display": "none"},  # Hidden when no notifications
            ),
        ],
    )

app.layout = dbc.Container(
    [
        # Global notification store (persists across app phases)
        dcc.Store(id="app-notifications", data=[]),

        # Notification panel (Offcanvas)
        build_notification_panel(),

        dbc.Navbar(
            dbc.Container(
                [
                    dbc.NavbarBrand("Portfolio26 – Research & ML", className="ms-2"),
                    # Notification bell icon in the header (right side)
                    build_notification_bell(),
                ],
                fluid=True,
                style={"display": "flex", "alignItems": "center"},
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
# Notification System Callbacks
# -----------------------------------------------------------------------------

@callback(
    Output("notification-badge", "children"),
    Output("notification-badge", "style"),
    Output("notification-badge", "className"),
    Input("app-notifications", "data"),
)
def update_notification_badge(notifications):
    """
    Update the notification badge based on the current notifications.
    - Badge count = number of messages
    - Badge color: Red if any ERROR, Orange if only SKIP
    - Badge hidden when count = 0
    """
    if not notifications:
        return "0", {"display": "none"}, "notification-badge notification-badge-skip"

    count = len(notifications)

    # Check if any notification is ERROR level
    has_error = any(n.get("level") == "ERROR" for n in notifications)

    if has_error:
        badge_class = "notification-badge notification-badge-error"
    else:
        badge_class = "notification-badge notification-badge-skip"

    return str(count), {"display": "flex"}, badge_class


@callback(
    Output("notification-panel", "is_open"),
    Input("notification-bell-btn", "n_clicks"),
    State("notification-panel", "is_open"),
    prevent_initial_call=True,
)
def toggle_notification_panel(n_clicks, is_open):
    """
    Toggle the notification panel when the bell icon is clicked.
    """
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output("notification-list", "children"),
    Output("notification-panel-footer", "style"),
    Input("app-notifications", "data"),
)
def render_notification_list(notifications):
    """
    Render the list of notifications in the panel.
    Messages are displayed reverse-chronologically (newest first).
    """
    if not notifications:
        return (
            html.Div("No notifications", className="notification-empty"),
            {"display": "none"},
        )

    # Sort by timestamp descending (newest first)
    sorted_notifications = sorted(
        notifications,
        key=lambda x: x.get("timestamp", ""),
        reverse=True,
    )

    items = []
    for notif in sorted_notifications:
        notif_id = notif.get("id", "")
        level = notif.get("level", "SKIP")
        timestamp = notif.get("timestamp", "")
        text = notif.get("text", "")

        # Format timestamp for display (show time portion)
        display_time = timestamp
        try:
            if "T" in timestamp:
                display_time = timestamp.split("T")[1][:8]  # HH:MM:SS
        except (IndexError, AttributeError):
            display_time = timestamp  # Fallback to original

        # CSS class based on level
        level_class = "notif-error" if level == "ERROR" else "notif-skip"

        item = html.Div(
            [
                html.Span(display_time, className="notification-timestamp"),
                html.Div(text, className="notification-text"),
                dbc.Button(
                    "×",
                    id={"type": "notification-dismiss-btn", "index": notif_id},
                    className="notification-dismiss-btn btn-close",
                    size="sm",
                    color="light",
                    outline=True,
                    title="Dismiss notification",
                    #**{"aria-label": "Dismiss notification"},
                ),
            ],
            className=f"notification-item {level_class}",
        )
        items.append(item)

    return items, {"display": "block"}


@callback(
    Output("app-notifications", "data", allow_duplicate=True),
    Input({"type": "notification-dismiss-btn", "index": ALL}, "n_clicks"),
    State("app-notifications", "data"),
    prevent_initial_call=True,
)
def dismiss_single_notification(n_clicks_list, notifications):
    """
    Dismiss a single notification when its dismiss button is clicked.
    """
    if not notifications:
        return []

    # Find which button was clicked
    triggered = ctx.triggered_id
    if triggered is None or not isinstance(triggered, dict):
        return notifications

    notif_id = triggered.get("index")
    if not notif_id:
        return notifications

    # Check if any button was actually clicked
    if not any(n for n in n_clicks_list if n):
        return notifications

    # Remove the notification with the matching id
    updated = [n for n in notifications if n.get("id") != notif_id]
    return updated


@callback(
    Output("app-notifications", "data", allow_duplicate=True),
    Input("notification-dismiss-all-btn", "n_clicks"),
    prevent_initial_call=True,
)
def dismiss_all_notifications(n_clicks):
    """
    Clear all notifications when Dismiss All is clicked.
    """
    if n_clicks:
        return []
    from dash import no_update
    return no_update



# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=PORT, host="127.0.0.1")
