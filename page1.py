# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:41:18 2025

@author: mauro

page1.py
"""

import os
import numpy as np
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, ctx
from datetime import datetime
import plotly.graph_objs as go
import re

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback
import pandas as pd

#--- Shared core module (layout and loader and strategy list)
from core import sh_layout


#--- Shared Layout import
from core.sh_layout import build_data_input_section, build_strategy_sidebar,strategy_color_for_uid
# --------------------------------------------------------------------
# Strategy & portfolio registry – JSON-based (core.registry)
# --------------------------------------------------------------------
from core.registry import (
    load_registry,
    add_or_update_strategy,
    list_strategies,
    list_portfolios,
    get_portfolio,
    add_or_update_portfolio,
)


# --------------------------------------------------------------------
# Rolling-analysis thresholds (Phase 1 defaults).
# These should eventually move to the global app settings.
# --------------------------------------------------------------------
ROLLING_MIN_TRADES_SHARPE = 15
ROLLING_MIN_TRADES_R = 10
ROLLING_MIN_TRADES_WIN = 10
ROLLING_MIN_TRADES_TAIL = 10
ROLLING_MIN_TRADES_DD = 10

# --------------------------------------------------------------------
# Overview metrics – initial equity & time constants - to be moved to settings
# --------------------------------------------------------------------
P1_INITIAL_EQUITY = 100_000.0          # used for CAGR/MAR/Sharpe (ann.)
TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25

# --------------- TIME CONSTANTS FOR PARAMETERS - MEAN R ENTRY TIME BUCKER SLIDERS
TIME_MARKS = {
    570: "09:30",
    600: "10:00",
    630: "10:30",
    660: "11:00",
    690: "11:30",
    720: "12:00",
    750: "12:30",
    780: "13:00",
    810: "13:30",
    840: "14:00",
    870: "14:30",
    900: "15:00",
    930: "15:30",
}

def _minutes_to_hhmm(m: float) -> str:
    """Convert minutes-from-midnight into 'HH:MM' string."""
    m_int = int(round(m))
    h = m_int // 60
    mm = m_int % 60
    return f"{h:02d}:{mm:02d}"

def _clean_name(raw: str) -> str: #To be removed as we no longer use the FOLD and SINGLE prefixes CHANGE REQUEST
    """
    Remove noisy prefixes like 'FOLD.' / 'SINGLE.' and extra spaces
    for display in legends / tooltips.
    """
    if not raw:
        return ""
    name = re.sub(r"^(FOLD\.|SINGLE\.)\s*", "", str(raw)).strip()
    return name or str(raw)

# --------------- VIX THRESHOLDS FOR PARAMETERS - MEAN R  VIX SLIDERS

VIX_SLIDER_MIN = 10
VIX_SLIDER_MAX = 45

VIX_MARKS = {
    10: "10",
    12: "12",
    13: "13",
    14: "14",
    15: "15",
    16: "16",
    17: "17",
    18: "18",
    19: "19",
    20: "20",
    21: "21",
    23: "23",
    25: "25",
    28: "28",
    30: "30",
    33: "33",
    37: "37",
    40: "40",
    45: "45",
}


def build_phase1_right_panel():
    """
    Phase 1 right-hand analytics panel (Strategy Analytics card with tabs).
    Does NOT include loader or strategy list. Reused later by main.py.
    """
    return dbc.Card(
        [
            dbc.CardHeader("Strategy Analytics"),
            dbc.Card(
                [
                    dcc.Tabs(
                        id="p1-analysis-tabs",
                        value="overview",
                        style={"backgroundColor": "#222222"},
                        children=[
                            dcc.Tab(
                                label="Overview",
                                value="overview",
                                style={
                                    "backgroundColor": "#333333",
                                    "color": "#BBBBBB",
                                    "padding": "8px 12px",
                                },
                                selected_style={
                                    "backgroundColor": "#111111",
                                    "color": "#FFFFFF",
                                    "padding": "8px 12px",
                                    "borderTop": "2px solid #1f77b4",
                                },
                                children=[
                                    # Metrics panel (cards) – filled by callback
                                    html.Div(
                                        id="p1-strategy-metrics",
                                        style={"fontSize": "0.85rem", "marginBottom": "0.75rem"},
                                    ),
                                
                                    # Mode toggle: cumulative vs all-selected
                                    html.Div(
                                        [
                                            html.Span(
                                                "Display mode:",
                                                style={"marginRight": "0.5rem", "fontSize": "0.85rem"},
                                            ),
                                            html.Div(
                                                dbc.RadioItems(
                                                    id="p1-overview-mode",
                                                    options=[
                                                        {"label": "Cumulative", "value": "cumulative"},
                                                        {"label": "All selected", "value": "individual"},
                                                    ],
                                                    value="cumulative",
                                                    inline=True,
                                                    className="btn-group",
                                                    inputClassName="btn-check",
                                                    labelClassName="btn btn-sm btn-outline-info",
                                                    labelCheckedClassName="btn btn-sm btn-info active",
                                                ),
                                            ),
                                        ],
                                        style={
                                            "marginBottom": "0.5rem",
                                            "display": "flex",
                                            "alignItems": "center",
                                            "gap": "0.75rem",
                                        },
                                    ),
                                
                                    # Equity and DD charts
                                    dcc.Graph(
                                        id="p1-equity-graph",
                                        figure={
                                            "data": [],
                                            "layout": {
                                                "template": "plotly_dark",
                                                "paper_bgcolor": "#222222",
                                                "plot_bgcolor": "#222222",
                                                "font": {"color": "#EEEEEE"},
                                            },
                                        },
                                        style={"height": "300px"},
                                    ),
                                    dcc.Graph(
                                        id="p1-dd-graph",
                                        figure={
                                            "data": [],
                                            "layout": {
                                                "template": "plotly_dark",
                                                "paper_bgcolor": "#222222",
                                                "plot_bgcolor": "#222222",
                                                "font": {"color": "#EEEEEE"},
                                            },
                                        },
                                        style={"height": "250px", "marginTop": "0.5rem"},
                                    ),
                                
                                    # Monthly returns heatmap
                                    dcc.Graph(
                                        id="p1-monthly-heatmap",
                                        figure={
                                            "data": [],
                                            "layout": {
                                                "template": "plotly_dark",
                                                "paper_bgcolor": "#222222",
                                                "plot_bgcolor": "#222222",
                                                "font": {"color": "#EEEEEE"},
                                            },
                                        },
                                        style={"height": "260px", "marginTop": "0.75rem"},
                                    ),
                                
                                    # P&L distribution histogram
                                    dcc.Graph(
                                        id="p1-pnl-histogram",
                                        figure={
                                            "data": [],
                                            "layout": {
                                                "template": "plotly_dark",
                                                "paper_bgcolor": "#222222",
                                                "plot_bgcolor": "#222222",
                                                "font": {"color": "#EEEEEE"},
                                            },
                                        },
                                        style={"height": "260px", "marginTop": "0.75rem"},
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="Subperiods",
                                value="subperiods",
                                style={
                                    "backgroundColor": "#333333",
                                    "color": "#BBBBBB",
                                    "padding": "8px 12px",
                                },
                                selected_style={
                                    "backgroundColor": "#111111",
                                    "color": "#FFFFFF",
                                    "padding": "8px 12px",
                                    "borderTop": "2px solid #1f77b4",
                                },
                                children=[
                                    html.Div(
                                        [
                                            # Controls row: switch between $ and % view
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "Monthly performance metric:",
                                                        style={
                                                            "marginRight": "0.5rem",
                                                            "fontSize": "0.85rem",
                                                        },
                                                    ),
                                                    dbc.RadioItems(
                                                        id="p1-subperiods-metric-mode",
                                                        options=[
                                                            {
                                                                "label": "Mean monthly P&L ($)",
                                                                "value": "pnl",
                                                            },
                                                            {
                                                                "label": "Mean monthly return (%)",
                                                                "value": "ret",
                                                            },
                                                        ],
                                                        value="pnl",
                                                        inline=True,
                                                        className="btn-group",
                                                        inputClassName="btn-check",
                                                        labelClassName="btn btn-sm btn-outline-info",
                                                        labelCheckedClassName="btn btn-sm btn-info active",
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "marginTop": "0.75rem",
                                                    "marginBottom": "0.5rem",
                                                },
                                            ),
                            
                                            # Row 1: mean monthly + Max DD (TOP)
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-subperiods-mean-monthly-graph",
                                                            config={"displayModeBar": False},
                                                        ),
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-subperiods-dd-graph",
                                                            config={"displayModeBar": False},
                                                        ),
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                            
                                            # Row 2: Win rate + Sharpe (BOTTOM)
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-subperiods-winrate-graph",
                                                            config={"displayModeBar": False},
                                                        ),
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-subperiods-sharpe-graph",
                                                            config={"displayModeBar": False},
                                                        ),
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                            
                                            # Row 3: equity curves per year
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-subperiods-equity-graph",
                                                            config={"displayModeBar": False},
                                                        ),
                                                        md=12,
                                                    ),
                                                ]
                                            ),
                                        ],
                                        style={
                                            "padding": "0.75rem 0.5rem",
                                        },
                                    )
                                ],
                            ),
                            dcc.Tab(
                                label="Rolling",
                                value="rolling",
                                style={
                                    "backgroundColor": "#333333",
                                    "color": "#BBBBBB",
                                    "padding": "8px 12px",
                                },
                                selected_style={
                                    "backgroundColor": "#111111",
                                    "color": "#FFFFFF",
                                    "padding": "8px 12px",
                                    "borderTop": "2px solid #1f77b4",
                                },
                                children=[
                                    html.Div(
                                        [
                                            # Controls row: rolling window selector
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "Rolling window (days):",
                                                        style={
                                                            "marginRight": "0.5rem",
                                                            "fontSize": "0.85rem",
                                                        },
                                                    ),
                                                    html.Div(
                                                        dbc.RadioItems(
                                                            id="p1-rolling-window",
                                                            options=[
                                                                {"label": "30d", "value": 30},
                                                                {"label": "60d", "value": 60},
                                                                {"label": "90d", "value": 90},
                                                                {"label": "120d", "value": 120},
                                                            ],
                                                            value=90,
                                                            inline=True,
                                                            className="btn-group",
                                                            inputClassName="btn-check",
                                                            labelClassName="btn btn-sm btn-outline-info",
                                                            labelCheckedClassName="btn btn-sm btn-info active",
                                                        ),
                                                        style={
                                                            "marginLeft": "0.75rem",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "marginTop": "0.75rem",
                                                    "marginBottom": "0.5rem",
                                                },
                                            ),

                            
                                            # Row 1: Rolling Sharpe + Rolling DD
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-rolling-sharpe-graph",
                                                            config={"displayModeBar": False},
                                                        ),
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-rolling-dd-graph",
                                                            config={"displayModeBar": False},
                                                        ),
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                            
                                            # Row 2: Rolling mean R + Rolling win rate
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-rolling-r-graph",
                                                            config={"displayModeBar": False},
                                                        ),
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-rolling-winrate-graph",
                                                            config={"displayModeBar": False},
                                                        ),
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                            
                                            # Row 3: Rolling tail loss (5% quantile)
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-rolling-tail-graph",
                                                            config={"displayModeBar": False},
                                                        ),
                                                        md=12,
                                                    ),
                                                ]
                                            ),
                                        ],
                                        style={
                                            "padding": "0.75rem 0.5rem",
                                        },
                                    )
                                ],
                            ),

                            dcc.Tab(
                                label="Tail Risk",
                                value="tail",
                                style={
                                    "backgroundColor": "#333333",
                                    "color": "#BBBBBB",
                                    "padding": "8px 12px",
                                },
                                selected_style={
                                    "backgroundColor": "#111111",
                                    "color": "#FFFFFF",
                                    "padding": "8px 12px",
                                    "borderTop": "2px solid #1f77b4",
                                },
                                children=[
                                    html.Div(
                                        [
                                            # Title
                                            html.H5(
                                                "Tail Risk & Fragility",
                                                style={"marginBottom": "0.75rem"},
                                            ),

                                            # Metrics cards container - filled by callback
                                            html.Div(
                                                id="p1-tail-metrics",
                                                style={
                                                    "fontSize": "0.85rem",
                                                    "marginBottom": "0.75rem",
                                                },
                                            ),

                                            html.Hr(
                                                style={
                                                    "marginTop": "0.5rem",
                                                    "marginBottom": "0.75rem",
                                                }
                                            ),

                                            # Controls for conditional horizon
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "Conditional horizon (trades after tail event):",
                                                        style={
                                                            "marginRight": "0.5rem",
                                                            "fontSize": "0.85rem",
                                                        },
                                                    ),
                                                    # Styled segmented buttons (5 / 10 / 20)
                                                    dbc.RadioItems(
                                                        id="p1-tail-horizon",
                                                        options=[
                                                            {"label": "5", "value": 5},
                                                            {"label": "10", "value": 10},
                                                            {"label": "20", "value": 20},
                                                        ],
                                                        value=10,
                                                        inline=True,
                                                        className="btn-group",
                                                        inputClassName="btn-check",
                                                        labelClassName="btn btn-sm btn-outline-info",
                                                        labelCheckedClassName="btn btn-sm btn-info active",
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "marginBottom": "0.75rem",
                                                },
                                            ),

                                            # Charts 2x2
                                            dbc.Row(
                                                [
                                                    # Top-left: log-scale histogram
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-tail-hist",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "xaxis": {"visible": False},
                                                                    "yaxis": {"visible": False},
                                                                    "annotations": [
                                                                        {
                                                                            "text": "No data",
                                                                            "xref": "paper",
                                                                            "yref": "paper",
                                                                            "showarrow": False,
                                                                            "font": {"color": "#AAAAAA"},
                                                                        }
                                                                    ],
                                                                },
                                                            },
                                                        ),
                                                        width=6,
                                                    ),

                                                    # Top-right: loss concentration chart
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-tail-lossconcentration",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "xaxis": {"visible": False},
                                                                    "yaxis": {"visible": False},
                                                                    "annotations": [
                                                                        {
                                                                            "text": "No data",
                                                                            "xref": "paper",
                                                                            "yref": "paper",
                                                                            "showarrow": False,
                                                                            "font": {"color": "#AAAAAA"},
                                                                        }
                                                                    ],
                                                                },
                                                            },
                                                        ),
                                                        width=6,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),

                                            dbc.Row(
                                                [
                                                    # Bottom-left: DD overlay of worst 3 periods
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-tail-ddoverlay",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "xaxis": {"visible": False},
                                                                    "yaxis": {"visible": False},
                                                                    "annotations": [
                                                                        {
                                                                            "text": "No data",
                                                                            "xref": "paper",
                                                                            "yref": "paper",
                                                                            "showarrow": False,
                                                                            "font": {"color": "#AAAAAA"},
                                                                        }
                                                                    ],
                                                                },
                                                            },
                                                        ),
                                                        width=6,
                                                    ),

                                                    # Bottom-right: large-loss conditional return
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-tail-conditional",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "xaxis": {"visible": False},
                                                                    "yaxis": {"visible": False},
                                                                    "annotations": [
                                                                        {
                                                                            "text": "No data",
                                                                            "xref": "paper",
                                                                            "yref": "paper",
                                                                            "showarrow": False,
                                                                            "font": {"color": "#AAAAAA"},
                                                                        }
                                                                    ],
                                                                },
                                                            },
                                                        ),
                                                        width=6,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),
                                        ],
                                        style={"padding": "0.75rem"},
                                    )
                                ],
                            ),

                            dcc.Tab(
                                label="Parameters",
                                value="params",
                                style={
                                    "backgroundColor": "#333333",
                                    "color": "#BBBBBB",
                                    "padding": "8px 12px",
                                },
                                selected_style={
                                    "backgroundColor": "#111111",
                                    "color": "#FFFFFF",
                                    "padding": "8px 12px",
                                    "borderTop": "2px solid #1f77b4",
                                },
                                children=[
                                    html.Div(
                                        [
                                            html.H5(
                                                "Parameter sensitivity",
                                                style={
                                                    "fontSize": "0.95rem",
                                                    "marginBottom": "0.5rem",
                                                },
                                            ),
                                            html.P(
                                                "How trade outcomes vary across basic parameters "
                                                "(risk size, time of day, and premium). "
                                                "All charts are based on per-trade R-multiples "
                                                "where Margin Req. is available.",
                                                style={
                                                    "fontSize": "0.8rem",
                                                    "color": "#AAAAAA",
                                                    "marginBottom": "0.75rem",
                                                },
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                html.Span(
                                                                    "Entry-time bucketing:",
                                                                    style={"marginRight": "0.75rem", "fontSize": "0.85rem"},
                                                                ),
                                            
                                                                dbc.ButtonGroup(
                                                                    [
                                                                        dbc.Button(
                                                                            "Auto",
                                                                            id="p1-param-entry-mode-auto",
                                                                            color="primary",
                                                                            outline=False,
                                                                            size="sm",
                                                                        ),
                                                                        dbc.Button(
                                                                            "Manual",
                                                                            id="p1-param-entry-mode-manual",
                                                                            color="primary",
                                                                            outline=True,
                                                                            size="sm",
                                                                        ),
                                                                    ],
                                                                    style={"marginRight": "1.5rem"},
                                                                ),
                                            
                                                                # Sliders container (initially hidden)
                                                                html.Div(
                                                                    id="p1-entry-manual-container",
                                                                    style={"display": "none"},
                                                                    children=[
                                                                        html.Div("Manual time boundaries (HH:MM):"),
                                            
                                                                        dcc.Slider(
                                                                            id="p1-entry-slider-1",
                                                                            min=570, max=930, step=5, value=660,
                                                                            marks=TIME_MARKS,
                                                                            tooltip={"always_visible": False, "placement": "top"},
                                                                        ),
                                                                        dcc.Slider(
                                                                            id="p1-entry-slider-2",
                                                                            min=570, max=930, step=5, value=780,
                                                                            marks=TIME_MARKS,
                                                                            tooltip={"always_visible": False, "placement": "top"},
                                                                        ),
                                                                        dcc.Slider(
                                                                            id="p1-entry-slider-3",
                                                                            min=570, max=930, step=5, value=870,
                                                                            marks=TIME_MARKS,
                                                                            tooltip={"always_visible": False, "placement": "top"},
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        )
                                                    )
                                                ],
                                                className="mb-3",
                                            ),

                                            # 2x2 grid of parameter charts
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-params-margin-bucket",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "xaxis": {"visible": False},
                                                                    "yaxis": {"visible": False},
                                                                    "annotations": [
                                                                        {
                                                                            "text": "No data",
                                                                            "xref": "paper",
                                                                            "yref": "paper",
                                                                            "x": 0.5,
                                                                            "y": 0.5,
                                                                            "showarrow": False,
                                                                            "font": {
                                                                                "color": "#AAAAAA"
                                                                            },
                                                                        }
                                                                    ],
                                                                },
                                                            },
                                                        ),
                                                        md=6,
                                                        width=12,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-params-time-bucket",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "xaxis": {"visible": False},
                                                                    "yaxis": {"visible": False},
                                                                    "annotations": [
                                                                        {
                                                                            "text": "No data",
                                                                            "xref": "paper",
                                                                            "yref": "paper",
                                                                            "x": 0.5,
                                                                            "y": 0.5,
                                                                            "showarrow": False,
                                                                            "font": {
                                                                                "color": "#AAAAAA"
                                                                            },
                                                                        }
                                                                    ],
                                                                },
                                                            },
                                                        ),
                                                        md=6,
                                                        width=12,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-params-premium-credit",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "xaxis": {"visible": False},
                                                                    "yaxis": {"visible": False},
                                                                    "annotations": [
                                                                        {
                                                                            "text": "No data",
                                                                            "xref": "paper",
                                                                            "yref": "paper",
                                                                            "x": 0.5,
                                                                            "y": 0.5,
                                                                            "showarrow": False,
                                                                            "font": {
                                                                                "color": "#AAAAAA"
                                                                            },
                                                                        }
                                                                    ],
                                                                },
                                                            },
                                                        ),
                                                        md=6,
                                                        width=12,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-params-premium-debit",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "xaxis": {"visible": False},
                                                                    "yaxis": {"visible": False},
                                                                    "annotations": [
                                                                        {
                                                                            "text": "No data",
                                                                            "xref": "paper",
                                                                            "yref": "paper",
                                                                            "x": 0.5,
                                                                            "y": 0.5,
                                                                            "showarrow": False,
                                                                            "font": {
                                                                                "color": "#AAAAAA"
                                                                            },
                                                                        }
                                                                    ],
                                                                },
                                                            },
                                                        ),
                                                        md=6,
                                                        width=12,
                                                    ),
                                                ]
                                            ),
                                            
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-params-dte",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "xaxis": {"visible": False},
                                                                    "yaxis": {"visible": False},
                                                                    "annotations": [
                                                                        {
                                                                            "text": "No data",
                                                                            "xref": "paper",
                                                                            "yref": "paper",
                                                                            "x": 0.5,
                                                                            "y": 0.5,
                                                                            "showarrow": False,
                                                                            "font": {
                                                                                "color": "#AAAAAA"
                                                                            },
                                                                        }
                                                                    ],
                                                                },
                                                            },
                                                        ),
                                                        md=4,
                                                        width=12,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-params-gap",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "xaxis": {"visible": False},
                                                                    "yaxis": {"visible": False},
                                                                    "annotations": [
                                                                        {
                                                                            "text": "No data",
                                                                            "xref": "paper",
                                                                            "yref": "paper",
                                                                            "x": 0.5,
                                                                            "y": 0.5,
                                                                            "showarrow": False,
                                                                            "font": {
                                                                                "color": "#AAAAAA"
                                                                            },
                                                                        }
                                                                    ],
                                                                },
                                                            },
                                                        ),
                                                        md=4,
                                                        width=12,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-params-move",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "xaxis": {"visible": False},
                                                                    "yaxis": {"visible": False},
                                                                    "annotations": [
                                                                        {
                                                                            "text": "No data",
                                                                            "xref": "paper",
                                                                            "yref": "paper",
                                                                            "x": 0.5,
                                                                            "y": 0.5,
                                                                            "showarrow": False,
                                                                            "font": {
                                                                                "color": "#AAAAAA"
                                                                            },
                                                                        }
                                                                    ],
                                                                },
                                                            },
                                                        ),
                                                        md=4,
                                                        width=12,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),

                                        ],
                                        style={"padding": "0.75rem 0.5rem"},
                                    )
                                ],
                            ),

                            dcc.Tab(
                                label="Regimes",
                                value="regimes",
                                style={
                                    "backgroundColor": "#333333",
                                    "color": "#BBBBBB",
                                    "padding": "8px 12px",
                                },
                                selected_style={
                                    "backgroundColor": "#111111",
                                    "color": "#FFFFFF",
                                    "padding": "8px 12px",
                                    "borderTop": "2px solid #1f77b4",
                                },
                                children=[
                                    html.Div(
                                        [
                                            html.H5(
                                                "Regime-based performance",
                                                style={
                                                    "fontSize": "0.95rem",
                                                    "marginBottom": "0.5rem",
                                                },
                                            ),
                                            html.P(
                                                "How per-trade R-multiples behave under different volatility and gap regimes. "
                                                "All charts below use trades from the selected strategies and, where possible, "
                                                "per-trade R = P/L divided by Margin Req.",
                                                style={
                                                    "fontSize": "0.8rem",
                                                    "color": "#AAAAAA",
                                                    "marginBottom": "0.75rem",
                                                },
                                            ),
                                            # Opening VIX bucketing controls (AUTO / MANUAL)
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                html.Span(
                                                                    "Opening VIX bucketing:",
                                                                    style={
                                                                        "marginRight": "0.75rem",
                                                                        "fontSize": "0.85rem",
                                                                    },
                                                                ),
                                                                dbc.ButtonGroup(
                                                                    [
                                                                        dbc.Button(
                                                                            "Auto",
                                                                            id="p1-regimes-vix-mode-auto",
                                                                            color="primary",
                                                                            outline=False,
                                                                            size="sm",
                                                                        ),
                                                                        dbc.Button(
                                                                            "Manual",
                                                                            id="p1-regimes-vix-mode-manual",
                                                                            color="primary",
                                                                            outline=True,
                                                                            size="sm",
                                                                        ),
                                                                    ],
                                                                    style={"marginRight": "1.5rem"},
                                                                ),
                                                                # Sliders container (initially hidden)
                                                                html.Div(
                                                                    id="p1-regimes-vix-manual-container",
                                                                    style={"display": "none"},
                                                                    children=[
                                                                        html.Div(
                                                                            "Manual VIX boundaries:",
                                                                            style={"marginTop": "0.5rem"},
                                                                        ),
                                                                        dcc.Slider(
                                                                            id="p1-regimes-vix-slider-1",
                                                                            min=VIX_SLIDER_MIN,
                                                                            max=VIX_SLIDER_MAX,
                                                                            step=1,
                                                                            value=15,
                                                                            marks=VIX_MARKS,
                                                                            tooltip={
                                                                                "always_visible": False,
                                                                                "placement": "top",
                                                                            },
                                                                            updatemode="mouseup",
                                                                        ),
                                                                        dcc.Slider(
                                                                            id="p1-regimes-vix-slider-2",
                                                                            min=VIX_SLIDER_MIN,
                                                                            max=VIX_SLIDER_MAX,
                                                                            step=1,
                                                                            value=20,
                                                                            marks=VIX_MARKS,
                                                                            tooltip={
                                                                                "always_visible": False,
                                                                                "placement": "top",
                                                                            },
                                                                            updatemode="mouseup",
                                                                        ),
                                                                        dcc.Slider(
                                                                            id="p1-regimes-vix-slider-3",
                                                                            min=VIX_SLIDER_MIN,
                                                                            max=VIX_SLIDER_MAX,
                                                                            step=1,
                                                                            value=25,
                                                                            marks=VIX_MARKS,
                                                                            tooltip={
                                                                                "always_visible": False,
                                                                                "placement": "top",
                                                                            },
                                                                            updatemode="mouseup",
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                        width=12,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),

                                            # VIX level at entry
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-regimes-vix-box",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "font": {"color": "#EEEEEE"},
                                                                    "title": {
                                                                        "text": "R distribution by Opening VIX regime",
                                                                        "x": 0.01,
                                                                        "xanchor": "left",
                                                                    },
                                                                },
                                                            },
                                                            style={"height": "300px"},
                                                        ),
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-regimes-vix-bar",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "font": {"color": "#EEEEEE"},
                                                                    "title": {
                                                                        "text": "Mean R by Opening VIX regime",
                                                                        "x": 0.01,
                                                                        "xanchor": "left",
                                                                    },
                                                                },
                                                            },
                                                            style={"height": "300px"},
                                                        ),
                                                        width=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # VIX trend while trade is open
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-regimes-trend-box",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "font": {"color": "#EEEEEE"},
                                                                    "title": {
                                                                        "text": "R distribution by VIX trend while in trade",
                                                                        "x": 0.01,
                                                                        "xanchor": "left",
                                                                    },
                                                                },
                                                            },
                                                            style={"height": "300px"},
                                                        ),
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-regimes-trend-bar",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "font": {"color": "#EEEEEE"},
                                                                    "title": {
                                                                        "text": "Mean R by VIX trend while in trade",
                                                                        "x": 0.01,
                                                                        "xanchor": "left",
                                                                    },
                                                                },
                                                            },
                                                            style={"height": "300px"},
                                                        ),
                                                        width=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Gap regimes
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-regimes-gap-box",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "font": {"color": "#EEEEEE"},
                                                                    "title": {
                                                                        "text": "R distribution by Gap regime",
                                                                        "x": 0.01,
                                                                        "xanchor": "left",
                                                                    },
                                                                },
                                                            },
                                                            style={"height": "300px"},
                                                        ),
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-regimes-gap-bar",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "font": {"color": "#EEEEEE"},
                                                                    "title": {
                                                                        "text": "Mean R by Gap regime",
                                                                        "x": 0.01,
                                                                        "xanchor": "left",
                                                                    },
                                                                },
                                                            },
                                                            style={"height": "300px"},
                                                        ),
                                                        width=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                        ],
                                        style={"padding": "0.75rem 0.5rem"},
                                    )
                                ],

                            ),
                            dcc.Tab(
                                label="Overfitting",
                                value="overfit",
                                style={
                                    "backgroundColor": "#333333",
                                    "color": "#BBBBBB",
                                    "padding": "8px 12px",
                                },
                                selected_style={
                                    "backgroundColor": "#111111",
                                    "color": "#FFFFFF",
                                    "padding": "8px 12px",
                                    "borderTop": "2px solid #1f77b4",
                                },
                                children=[
                                    html.Div(
                                        [
                                            html.H5(
                                                "Overfitting diagnostics",
                                                style={
                                                    "fontSize": "0.95rem",
                                                    "marginBottom": "0.5rem",
                                                },
                                            ),
                                            html.P(
                                                "Bootstrap, top-trade removal, and Monte Carlo path bootstraps "
                                                "to assess robustness of the selected strategy (or mix). "
                                                "All tests work on per-trade outcomes using the chosen metric.",
                                                style={
                                                    "fontSize": "0.8rem",
                                                    "color": "#AAAAAA",
                                                    "marginBottom": "0.75rem",
                                                },
                                            ),
                                            # Controls row
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "Performance metric:",
                                                                        style={
                                                                            "marginRight": "0.5rem",
                                                                            "fontSize": "0.85rem",
                                                                        },
                                                                    ),
                                                                    dcc.RadioItems(
                                                                        id="p1-overfit-metric-mode",
                                                                        options=[
                                                                            {
                                                                                "label": "R-multiple",
                                                                                "value": "R",
                                                                            },
                                                                            {
                                                                                "label": "P&L ($)",
                                                                                "value": "PL",
                                                                            },
                                                                        ],
                                                                        value="R",
                                                                        inline=True,
                                                                        labelStyle={
                                                                            "marginRight": "0.75rem",
                                                                            "fontSize": "0.8rem",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={"marginBottom": "0.4rem"},
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "Bootstrap runs:",
                                                                        style={
                                                                            "marginRight": "0.25rem",
                                                                            "fontSize": "0.8rem",
                                                                        },
                                                                    ),
                                                                    dcc.Input(
                                                                        id="p1-overfit-bootstrap-n",
                                                                        type="number",
                                                                        min=100,
                                                                        max=5000,
                                                                        step=100,
                                                                        value=500,
                                                                        style={"width": "70px"},
                                                                    ),
                                                                    html.Span(
                                                                        "  Block length:",
                                                                        style={
                                                                            "marginLeft": "0.75rem",
                                                                            "marginRight": "0.25rem",
                                                                            "fontSize": "0.8rem",
                                                                        },
                                                                    ),
                                                                    dcc.Input(
                                                                        id="p1-overfit-bootstrap-block",
                                                                        type="number",
                                                                        min=1,
                                                                        max=50,
                                                                        step=1,
                                                                        value=1,
                                                                        style={"width": "60px"},
                                                                    ),
                                                                    dbc.Button(
                                                                        "Run bootstrap",
                                                                        id="p1-overfit-run-bootstrap",
                                                                        color="primary",
                                                                        size="sm",
                                                                        className="ms-3",
                                                                        n_clicks=0,
                                                                    ),
                                                                ],
                                                                style={"fontSize": "0.8rem"},
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "Top trades K to remove:",
                                                                        style={
                                                                            "marginRight": "0.5rem",
                                                                            "fontSize": "0.8rem",
                                                                        },
                                                                    ),
                                                                    dcc.Slider(
                                                                        id="p1-overfit-topk",
                                                                        min=0,
                                                                        max=10,
                                                                        step=1,
                                                                        value=3,
                                                                        marks={
                                                                            i: str(i)
                                                                            for i in range(0, 11)
                                                                        },
                                                                        tooltip={
                                                                            "always_visible": False,
                                                                            "placement": "top",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={"marginBottom": "0.5rem"},
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "MC permutations (bootstrap paths):",
                                                                        style={
                                                                            "marginRight": "0.25rem",
                                                                            "fontSize": "0.8rem",
                                                                        },
                                                                    ),
                                                                    dcc.Input(
                                                                        id="p1-overfit-mc-n",
                                                                        type="number",
                                                                        min=100,
                                                                        max=5000,
                                                                        step=100,
                                                                        value=500,
                                                                        style={"width": "70px"},
                                                                    ),
                                                                    dbc.Button(
                                                                        "Run MC",
                                                                        id="p1-overfit-run-mc",
                                                                        color="primary",
                                                                        size="sm",
                                                                        className="ms-3",
                                                                        n_clicks=0,
                                                                    ),
                                                                ],
                                                                style={"fontSize": "0.8rem"},
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Row 1: Bootstrap & Top-K
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-overfit-bootstrap-hist",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "font": {"color": "#EEEEEE"},
                                                                    "title": {
                                                                        "text": "Bootstrap Sharpe distribution",
                                                                        "x": 0.01,
                                                                        "xanchor": "left",
                                                                    },
                                                                },
                                                            },
                                                            style={"height": "300px"},
                                                        ),
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-overfit-topk-bar",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "font": {"color": "#EEEEEE"},
                                                                    "title": {
                                                                        "text": "Effect of removing top-K trades",
                                                                        "x": 0.01,
                                                                        "xanchor": "left",
                                                                    },
                                                                },
                                                            },
                                                            style={"height": "300px"},
                                                        ),
                                                        width=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Row 2: MC DD & final equity
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-overfit-mc-dd-hist",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "font": {"color": "#EEEEEE"},
                                                                    "title": {
                                                                        "text": "MC distribution of max drawdown",
                                                                        "x": 0.01,
                                                                        "xanchor": "left",
                                                                    },
                                                                },
                                                            },
                                                            style={"height": "300px"},
                                                        ),
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-overfit-mc-final-hist",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "font": {"color": "#EEEEEE"},
                                                                    "title": {
                                                                        "text": "MC distribution of final cumulative result",
                                                                        "x": 0.01,
                                                                        "xanchor": "left",
                                                                    },
                                                                },
                                                            },
                                                            style={"height": "300px"},
                                                        ),
                                                        width=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Row 3: MC equity fan chart
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="p1-overfit-mc-fan",
                                                            figure={
                                                                "data": [],
                                                                "layout": {
                                                                    "template": "plotly_dark",
                                                                    "paper_bgcolor": "#222222",
                                                                    "plot_bgcolor": "#222222",
                                                                    "font": {"color": "#EEEEEE"},
                                                                    "title": {
                                                                        "text": "MC bootstrapped equity paths",
                                                                        "x": 0.01,
                                                                        "xanchor": "left",
                                                                    },
                                                                },
                                                            },
                                                            style={"height": "360px"},
                                                        ),
                                                        width=12,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Row 4: metrics (3 blocks)
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                html.H6(
                                                                    "Bootstrap summary",
                                                                    style={
                                                                        "fontSize": "0.85rem",
                                                                        "marginTop": "0.25rem",
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    id="p1-overfit-metrics-bootstrap",
                                                                    style={"fontSize": "0.8rem"},
                                                                ),
                                                                html.H6(
                                                                    "Top-K summary",
                                                                    style={
                                                                        "fontSize": "0.85rem",
                                                                        "marginTop": "0.75rem",
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    id="p1-overfit-metrics-topk",
                                                                    style={"fontSize": "0.8rem"},
                                                                ),
                                                                html.H6(
                                                                    "MC summary",
                                                                    style={
                                                                        "fontSize": "0.85rem",
                                                                        "marginTop": "0.75rem",
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    id="p1-overfit-metrics-mc",
                                                                    style={"fontSize": "0.8rem"},
                                                                ),
                                                            ]
                                                        ),
                                                        width=12,
                                                    ),
                                                ]
                                            ),
                                        ],
                                        style={"padding": "0.75rem 0.5rem"},
                                    )
                                ],
                            ),
                        ],
                    )
                ]
            ),
        ]
    )




# --------------------------------------------------------------------
# Page 1 layout: Strategies R&D
# --------------------------------------------------------------------
def layout_page_1():
    """
    Layout for Phase 1 – Strategy R&D.

    Top: collapsible 'Data Input – Strategies' panel.
    Middle: left sidebar for strategy list (registry placeholder),
            right main area for analytics (charts/tables).
    """

    subfolders = sh_layout._list_immediate_subfolders(sh_layout.ROOT_DATA_DIR)

    if not subfolders:
        folder_info = html.Div(
            [
                html.Div(
                    f"No subfolders found under ROOT_DATA_DIR = '{sh_layout.ROOT_DATA_DIR}'.",
                    style={"color": "red"},
                ),
                html.Div(
                    "Create subfolders (e.g. Strat01, Strat02, ...) under that path "
                    "and restart the app.",
                    style={"fontSize": "0.9rem", "marginTop": "0.5rem"},
                ),
            ]
        )
        folder_checklist = html.Div()  # empty
    else:
        # Build checklist options from discovered subfolders
        options = [
            {"label": name, "value": full_path} for name, full_path in subfolders
        ]
        # DEFAULT: all folders selected
        default_values = [full_path for _, full_path in subfolders]

        folder_info = html.Div(
            [
                html.Div("Root data folder:", style={"fontSize": "0.85rem"}),
                html.Code(sh_layout.ROOT_DATA_DIR, style={"fontSize": "0.8rem"}),
                html.Div(
                    f"Found {len(subfolders)} subfolder(s).",
                    style={"fontSize": "0.85rem", "marginTop": "0.25rem"},
                ),
            ]
        )
        folder_checklist = dcc.Checklist(
            id="p1-folder-checklist",
            options=options,
            value=default_values,  # all selected by default
            labelStyle={"display": "block", "marginBottom": "0.15rem"},
            style={"maxHeight": "200px", "overflowY": "auto", "marginTop": "0.5rem"},
        )

        # Existing portfolios for the dropdown (if any)
        portfolios = list_portfolios()
        portfolio_options = [
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

    # ----------------- Data Input Card (inside collapse) -----------------
   
    return dbc.Container(
        [
            html.H3("Phase 1 – Strategy R&D", className="mb-3"),
            
            dcc.Store(id="p1-current-portfolio-id", data=None),

            # ----------------------------------------------------------------
            # Collapsible Data Input section
            # ----------------------------------------------------------------
            
            build_data_input_section(folder_info, folder_checklist, portfolio_options),
            
            html.Hr(),

            # ----------------------------------------------------------------
            # Main content: left = strategy list sidebar, right = analytics
            # ----------------------------------------------------------------
            dbc.Row(
                [
                    # LEFT: Strategy list / registry placeholder (sidebar)
                    dbc.Col(
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
                                            style={"fontSize": "0.8rem", "marginBottom": "0.4rem"},
                                        ),
                                
                                        dcc.Checklist(
                                            id="p1-strategy-checklist",
                                            options=[],
                                            value=[],
                                            labelStyle={"display": "block", "marginBottom": "0.15rem"},
                                            style={
                                                "maxHeight": "400px",
                                                "overflowY": "auto",
                                                "fontSize": "0.85rem",
                                            },
                                        ),
                                
                                        html.Div(
                                            id="p1-strategy-summary",
                                            children="No strategies loaded yet.",
                                            style={"fontSize": "0.8rem", "marginTop": "0.4rem", "color": "#AAAAAA"},
                                        ),
                                    ]
                                )

                            ],
                            style={"position": "sticky", "top": "80px"},
                        ),
                        width=3,
                    ),

                    # RIGHT: Strategy analytics area
                    # RIGHT: Phase 1 analytics panel (extracted helper)
                    dbc.Col(
                        build_phase1_right_panel(),
                        width=9,
                    ),
                ],
                className="mt-3",
            ),
        ],
        fluid=True,
        className="mt-3",
    )


def _get_strategy_trades(strategy_id: str):
    """
    Return a DataFrame with at least ['Date Closed', 'P/L'] for the given strategy_id.

    Priority:
    1) If p1_strategy_store has a cached 'df', use that (upgrading to include extra columns if possible).
    2) Otherwise, if registry has a file_path that exists, read from disk and cache.
    3) If nothing works, return an empty DataFrame.
    """

    # Columns we would like to have available for downstream analytics
    needed_extra = [
        "Margin Req.",
        "Date Opened",
        "Time Opened",
        "Premium",
        "Gap",
        "Movement",
        "Legs",
        "Opening VIX",
        "Closing VIX",
        "Opening Price",
        "Closing Price",
    ]

    # 1) Cached in-memory
    meta = sh_layout.p1_strategy_store.get(strategy_id)
    if meta is not None and isinstance(meta.get("df"), pd.DataFrame):
        df = meta["df"]

        # If any of the important extra columns are missing, try to reload from disk
        if any(col not in df.columns for col in needed_extra):
            file_path = meta.get("file_path")
            if file_path and os.path.isfile(file_path):
                try:
                    df_raw = pd.read_csv(file_path)
                    base_cols = ["Date Closed", "P/L"]
                    extra_cols = [c for c in needed_extra if c in df_raw.columns]
                    df = df_raw[base_cols + extra_cols].copy()
                    # Upgrade cache
                    meta["df"] = df
                    sh_layout.p1_strategy_store[strategy_id] = meta
                except Exception:
                    # If reload fails, keep existing df
                    pass
        return df.copy()

    # 2) Try reading from registry metadata
    #    list_strategies() returns all registry entries; we find the one matching id
    all_strats = list_strategies()
    row = None
    for s in all_strats:
        if s.get("id") == strategy_id:
            row = s
            break

    if row is not None:
        file_path = row.get("file_path")
        if file_path and file_path not in ("(uploaded)", "(unknown)"):
            try:
                df_raw = pd.read_csv(file_path)
                base_cols = ["Date Closed", "P/L"]
                extra_cols = [c for c in needed_extra if c in df_raw.columns]
                df = df_raw[base_cols + extra_cols].copy()
                # Cache it for next time
                meta = sh_layout.p1_strategy_store.get(strategy_id, {"id": strategy_id})
                meta["file_path"] = file_path
                meta["df"] = df
                sh_layout.p1_strategy_store[strategy_id] = meta
                return df.copy()
            except Exception:
                pass

    # 3) Nothing usable found
    return pd.DataFrame(columns=["Date Closed", "P/L"])



# --------------------------------------------------------------------
# Callbacks for Page 1
# --------------------------------------------------------------------


@callback(
    Output("p1-subperiods-mean-monthly-graph", "figure"),
    Output("p1-subperiods-dd-graph", "figure"),
    Output("p1-subperiods-winrate-graph", "figure"),
    Output("p1-subperiods-sharpe-graph", "figure"),
    Output("p1-subperiods-equity-graph", "figure"),
    Input("p1-strategy-checklist", "value"),
    Input("p1-subperiods-metric-mode", "value"),
)
def update_subperiods(selected_strategy_ids, metric_mode):
    """
    Subperiods section:
    - Aggregates all selected strategies (cumulative view).
    - Splits by calendar year (2022, 2023, 2024, 2025 YTD, ...).
    - Produces:
        * Mean monthly P&L / return per year (switchable),
        * Max DD per year (switchable $ / %),
        * Win rate per year,
        * Sharpe per year (monthly returns),
        * Per-year equity curves (rebased within year, actual dates on x-axis).
    """
    selected_strategy_ids = selected_strategy_ids or []
    metric_mode = metric_mode or "pnl"

    # Base empty figure
    empty_layout = {
        "template": "plotly_dark",
        "paper_bgcolor": "#222222",
        "plot_bgcolor": "#222222",
        "xaxis": {"visible": False},
        "yaxis": {"visible": False},
        "annotations": [
            {
                "text": "No data",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"color": "#888888"},
            }
        ],
    }
    empty_fig = {"data": [], "layout": empty_layout}

    if not selected_strategy_ids:
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    # Helper: green colormap for "good" metrics
    def make_green_colors(values):
        valid = [v for v in values if v is not None]
        if not valid:
            return ["#00AA00"] * len(values)
        vmin = min(valid)
        vmax = max(valid)
        colors = []
        for v in values:
            if v is None:
                colors.append("#00AA00")
                continue
            if vmax == vmin:
                t = 1.0
            else:
                t = (v - vmin) / (vmax - vmin)
            # light green (102,255,102) -> dark green (0,102,0)
            r = int(102 + (0 - 102) * t)
            g = int(255 + (102 - 255) * t)
            b = int(102 + (0 - 102) * t)
            colors.append(f"rgb({r},{g},{b})")
        return colors

    # Helper: red colormap for drawdowns (bigger |DD| = darker red)
    def make_red_colors(values):
        valid = [abs(v) for v in values if v is not None]
        if not valid:
            return ["#AA0000"] * len(values)
        vmin = min(valid)
        vmax = max(valid)
        colors = []
        for v in values:
            if v is None:
                colors.append("#AA0000")
                continue
            mag = abs(v)
            if vmax == vmin:
                t = 1.0
            else:
                t = (mag - vmin) / (vmax - vmin)
            # light red (255,153,153) -> dark red (153,0,0)
            r = int(255 + (153 - 255) * t)
            g = int(153 + (0 - 153) * t)
            b = int(153 + (0 - 153) * t)
            colors.append(f"rgb({r},{g},{b})")
        return colors

    # ----------------------------------------------------------
    # Collect trades for all selected strategies (cumulative view)
    # ----------------------------------------------------------
    frames = []
    for sid in selected_strategy_ids:
        df = _get_strategy_trades(sid)
        if df.empty:
            continue

        df = df.copy()
        df["Date Closed"] = pd.to_datetime(df["Date Closed"], errors="coerce")
        df = df.dropna(subset=["Date Closed"])

        df["P/L"] = pd.to_numeric(df["P/L"], errors="coerce")
        df = df.dropna(subset=["P/L"])

        frames.append(df)

    if not frames:
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    all_trades = pd.concat(frames, ignore_index=True)
    all_trades = all_trades.sort_values("Date Closed")

    # Add year and month columns
    all_trades["year"] = all_trades["Date Closed"].dt.year
    all_trades["month"] = all_trades["Date Closed"].dt.month

    years = sorted(all_trades["year"].unique())
    if not years:
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    # ----------------------------------------------------------
    # Monthly P&L and returns (100k baseline)
    # ----------------------------------------------------------
    monthly = (
        all_trades.groupby(["year", "month"])["P/L"]
        .sum()
        .reset_index()
    )
    monthly["monthly_return"] = monthly["P/L"] / 100000.0

    mean_pnl_by_year = monthly.groupby("year")["P/L"].mean()
    mean_ret_by_year = monthly.groupby("year")["monthly_return"].mean()

    current_year = datetime.now().year
    year_index = list(mean_pnl_by_year.index)
    year_labels = [
        f"{int(y)} (YTD)" if int(y) == current_year else f"{int(y)}"
        for y in year_index
    ]

    if metric_mode == "ret":
        y_values = (mean_ret_by_year.loc[year_index] * 100.0).values  # %
        y_title = "Average monthly return (%)"
    else:
        y_values = mean_pnl_by_year.loc[year_index].values
        y_title = "Average monthly P&L ($)"

    mean_colors = make_green_colors(list(y_values))

    mean_monthly_fig = go.Figure(
        data=[
            go.Bar(
                x=year_labels,
                y=y_values,
                marker={"color": mean_colors},
            )
        ],
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis={"title": "Year"},
            yaxis={"title": y_title},
        ),
    )

    # ----------------------------------------------------------
    # Sharpe per year (based on monthly returns)
    # ----------------------------------------------------------
    sharpe_by_year = {}
    for y in years:
        rets = monthly.loc[monthly["year"] == y, "monthly_return"]
        if len(rets) >= 2:
            mu = rets.mean()
            sigma = rets.std(ddof=1)
            if sigma > 0:
                sharpe = mu / sigma * (12 ** 0.5)
            else:
                sharpe = None
        else:
            sharpe = None
        sharpe_by_year[y] = sharpe

    sharpe_x = [
        f"{int(y)} (YTD)" if int(y) == current_year else f"{int(y)}"
        for y in years
    ]
    sharpe_y = [sharpe_by_year.get(y, None) for y in years]
    sharpe_colors = make_green_colors(sharpe_y)

    sharpe_fig = go.Figure(
        data=[
            go.Bar(
                x=sharpe_x,
                y=sharpe_y,
                marker={"color": sharpe_colors},
            )
        ],
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis={"title": "Year"},
            yaxis={
                "title": "Sharpe (monthly)",
                "zeroline": True,
                "zerolinecolor": "#555555",
            },
        ),
    )

    # ----------------------------------------------------------
    # Win rate per year (per trade)
    # ----------------------------------------------------------
    winrate_x = []
    winrate_y = []
    for y in years:
        df_y = all_trades[all_trades["year"] == y]
        n_trades_y = len(df_y)
        if n_trades_y > 0:
            win_rate = (df_y["P/L"] > 0).mean() * 100.0  # %
        else:
            win_rate = None
        label = f"{int(y)} (YTD)" if int(y) == current_year else f"{int(y)}"
        winrate_x.append(label)
        winrate_y.append(win_rate)

    winrate_colors = make_green_colors(winrate_y)

    winrate_fig = go.Figure(
        data=[
            go.Bar(
                x=winrate_x,
                y=winrate_y,
                marker={"color": winrate_colors},
            )
        ],
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis={"title": "Year"},
            yaxis={"title": "Win rate (%)", "range": [0, 100]},
        ),
    )

    # ----------------------------------------------------------
    # Daily equity & per-year max drawdown
    # ----------------------------------------------------------
    daily_pnl = (
        all_trades.groupby("Date Closed")["P/L"].sum().sort_index()
    )
    daily_equity = daily_pnl.cumsum()

    dd_labels = []
    dd_values = []
    equity_traces = []

    for y in years:
        mask = daily_equity.index.year == y
        if not mask.any():
            continue

        eq_y = daily_equity[mask]
        # Rebase equity for the year to start at 0
        eq_y_rebased = eq_y - eq_y.iloc[0]

        label = f"{int(y)} (YTD)" if int(y) == current_year else f"{int(y)}"

        # Equity trace for this year (actual dates on x-axis)
        equity_traces.append(
            go.Scatter(
                x=eq_y.index,
                y=eq_y_rebased.values,
                mode="lines",
                name=label,
            )
        )

        # Drawdown for this year
        cummax = eq_y_rebased.cummax()
        dd_series = eq_y_rebased - cummax
        max_dd_y = dd_series.min() if not dd_series.empty else 0.0

        dd_labels.append(label)
        dd_values.append(max_dd_y)

    # Switchable $ / % for DD as well
    if metric_mode == "ret":
        dd_plot_values = [
            v / 100000.0 * 100.0 if v is not None else None for v in dd_values
        ]
        dd_y_title = "Max drawdown per year (%)"
    else:
        dd_plot_values = dd_values
        dd_y_title = "Max drawdown per year ($)"

    dd_colors = make_red_colors(dd_plot_values)

    dd_fig = go.Figure(
        data=[
            go.Bar(
                x=dd_labels,
                y=dd_plot_values,
                marker={"color": dd_colors},
            )
        ],
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis={"title": "Year"},
            yaxis={"title": dd_y_title},
        ),
    )

    equity_fig = go.Figure(
        data=equity_traces,
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis={"title": "Date"},
            yaxis={"title": "Equity (rebased within year)"},
            legend={"orientation": "h", "y": -0.2},
        ),
    )

    # IMPORTANT: order matches the Outputs above
    return mean_monthly_fig, dd_fig, winrate_fig, sharpe_fig, equity_fig


@callback(
    Output("p1-rolling-sharpe-graph", "figure"),
    Output("p1-rolling-dd-graph", "figure"),
    Output("p1-rolling-r-graph", "figure"),
    Output("p1-rolling-winrate-graph", "figure"),
    Output("p1-rolling-tail-graph", "figure"),
    Input("p1-strategy-checklist", "value"),
    Input("p1-rolling-window", "value"),
)
def update_rolling(selected_strategy_ids, window_days):
    """
    Rolling robustness analysis (cumulative across selected strategies), using
    vectorised time-based rolling windows.

    - Rolling Sharpe (trade-level, 30/60/90/120 days)
    - Rolling max drawdown in window (daily equity)
    - Rolling mean R-multiple (P/L / Margin Req.)
    - Rolling win rate
    - Rolling tail loss (5% quantile of P/L)

    Thresholds for minimum number of trades per window are defined at module
    level (ROLLING_MIN_TRADES_*). These should move to app settings later.
    """

    selected_strategy_ids = selected_strategy_ids or []
    window_days = int(window_days) if window_days else 90
    window_str = f"{window_days}D"

    # Base empty figure
    empty_layout = {
        "template": "plotly_dark",
        "paper_bgcolor": "#222222",
        "plot_bgcolor": "#222222",
        "xaxis": {"visible": False},
        "yaxis": {"visible": False},
        "annotations": [
            {
                "text": "No data",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"color": "#888888"},
            }
        ],
    }
    empty_fig = {"data": [], "layout": empty_layout}

    if not selected_strategy_ids:
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    # ------------------------------------------------------------------
    # Collect trades for all selected strategies (cumulative view)
    # ------------------------------------------------------------------
    frames = []
    for sid in selected_strategy_ids:
        df = _get_strategy_trades(sid)
        if df is None or df.empty:
            continue

        df = df.copy()
        df["Date Closed"] = pd.to_datetime(df["Date Closed"], errors="coerce")
        df = df.dropna(subset=["Date Closed"])

        df["P/L"] = pd.to_numeric(df["P/L"], errors="coerce")

        # Margin Req. may or may not be present depending on the CSV
        if "Margin Req." in df.columns:
            df["Margin Req."] = pd.to_numeric(df["Margin Req."], errors="coerce")
        else:
            df["Margin Req."] = pd.NA

        df = df.dropna(subset=["P/L"])
        frames.append(df)

    if not frames:
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    all_trades = pd.concat(frames, ignore_index=True)
    all_trades = all_trades.sort_values("Date Closed")

    # Normalised calendar day (for daily aggregation)
    all_trades["day"] = all_trades["Date Closed"].dt.normalize()

    # R-multiple per trade where we have a valid margin requirement
    all_trades["Margin Req."] = pd.to_numeric(
        all_trades["Margin Req."], errors="coerce"
    )

    # Build R-multiple as a float series with NaNs (not pd.NA / object)
    margin_mask = all_trades["Margin Req."] > 0
    all_trades["R"] = np.nan
    all_trades.loc[margin_mask, "R"] = (
        all_trades.loc[margin_mask, "P/L"] / all_trades.loc[margin_mask, "Margin Req."]
    )
    # Ensure dtype is float, coerce anything weird to NaN
    all_trades["R"] = pd.to_numeric(all_trades["R"], errors="coerce")


    # ------------------------------------------------------------------
    # Trade-indexed rolling metrics (Sharpe, win rate, tail, mean R)
    # ------------------------------------------------------------------
    trades = all_trades.set_index("Date Closed").sort_index()
    pnl_series = trades["P/L"]

    # Number of trades in each rolling window (trade-level index)
    trade_count_roll = pnl_series.rolling(window_str).count()

    # Sharpe (trade-level)
    rolling_mean = pnl_series.rolling(window_str).mean()
    rolling_std = pnl_series.rolling(window_str).std(ddof=1)
    sharpe_series = rolling_mean / rolling_std * (252 ** 0.5)
    sharpe_series[rolling_std <= 0] = pd.NA
    sharpe_series[trade_count_roll < ROLLING_MIN_TRADES_SHARPE] = pd.NA

    # Win rate
    wins = (pnl_series > 0).astype(float)
    win_sum = wins.rolling(window_str).sum()
    win_rate_series = (win_sum / trade_count_roll) * 100.0
    win_rate_series[trade_count_roll < ROLLING_MIN_TRADES_WIN] = pd.NA

    # Tail loss (5% quantile of P/L)
    tail_series = pnl_series.rolling(window_str).quantile(0.05)
    tail_series[trade_count_roll < ROLLING_MIN_TRADES_TAIL] = pd.NA

    # Mean R per trade
    #r_series = trades["R"]
    r_series = trades["R"].astype(float)

    r_sum = r_series.rolling(window_str).sum()
    r_count = r_series.rolling(window_str).count()
    r_mean_series = r_sum / r_count
    r_mean_series[r_count < ROLLING_MIN_TRADES_R] = pd.NA

    # Collapse trade-indexed series to one point per calendar day
    sharpe_daily = sharpe_series.groupby(sharpe_series.index.normalize()).last()
    win_rate_daily = win_rate_series.groupby(win_rate_series.index.normalize()).last()
    tail_daily = tail_series.groupby(tail_series.index.normalize()).last()
    r_mean_daily = r_mean_series.groupby(r_mean_series.index.normalize()).last()

    # ------------------------------------------------------------------
    # Daily equity & rolling max drawdown in window
    # ------------------------------------------------------------------
    daily_pnl = all_trades.groupby("day")["P/L"].sum()
    daily_trade_count = all_trades.groupby("day")["P/L"].size()
    daily_eq = daily_pnl.cumsum()

    def _max_drawdown(series: pd.Series) -> float:
        if series.empty:
            return 0.0
        cummax = series.cummax()
        dd = series - cummax
        return float(dd.min())

    dd_daily = daily_eq.rolling(window_str).apply(_max_drawdown, raw=False)
    dd_trade_count_roll = daily_trade_count.rolling(window_str).sum()
    dd_daily[dd_trade_count_roll < ROLLING_MIN_TRADES_DD] = pd.NA

    # ------------------------------------------------------------------
    # Build figures (drop NaNs so we don't plot empty leading sections)
    # ------------------------------------------------------------------
    sharpe_daily = sharpe_daily.dropna()
    win_rate_daily = win_rate_daily.dropna()
    tail_daily = tail_daily.dropna()
    r_mean_daily = r_mean_daily.dropna()
    dd_daily = dd_daily.dropna()

    if (
        sharpe_daily.empty
        and win_rate_daily.empty
        and tail_daily.empty
        and r_mean_daily.empty
        and dd_daily.empty
    ):
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    # Sharpe figure
    sharpe_fig = go.Figure(
        data=[
            go.Scatter(
                x=sharpe_daily.index,
                y=sharpe_daily.values,
                mode="lines",
                name=f"Rolling Sharpe ({window_days}d)",
            )
        ],
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis={"title": "Date"},
            yaxis={
                "title": f"Sharpe ({window_days}d, trade-level)",
                "zeroline": True,
                "zerolinecolor": "#555555",
            },
        ),
    )

    # Rolling DD figure (daily equity)
    dd_fig = go.Figure(
        data=[
            go.Scatter(
                x=dd_daily.index,
                y=dd_daily.values,
                mode="lines",
                name=f"Rolling max DD ({window_days}d)",
            )
        ],
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis={"title": "Date"},
            yaxis={"title": f"Max drawdown in window (${window_days}d)"},
        ),
    )

    # Rolling mean R figure
    r_fig = go.Figure(
        data=[
            go.Scatter(
                x=r_mean_daily.index,
                y=r_mean_daily.values,
                mode="lines",
                name=f"Rolling mean R ({window_days}d)",
            )
        ],
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis={"title": "Date"},
            yaxis={"title": f"Mean R per trade ({window_days}d)"},
        ),
    )

    # Rolling win rate figure
    winrate_fig = go.Figure(
        data=[
            go.Scatter(
                x=win_rate_daily.index,
                y=win_rate_daily.values,
                mode="lines",
                name=f"Rolling win rate ({window_days}d)",
            )
        ],
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis={"title": "Date"},
            yaxis={"title": f"Win rate (%) ({window_days}d)", "range": [0, 100]},
        ),
    )

    # Rolling tail loss figure
    tail_fig = go.Figure(
        data=[
            go.Scatter(
                x=tail_daily.index,
                y=tail_daily.values,
                mode="lines",
                name=f"Rolling 5% tail P&L ({window_days}d)",
            )
        ],
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis={"title": "Date"},
            yaxis={"title": f"5% quantile P&L ({window_days}d)"},
        ),
    )

    return sharpe_fig, dd_fig, r_fig, winrate_fig, tail_fig




@callback(
    Output("p1-strategy-metrics", "children"),
    Output("p1-equity-graph", "figure"),
    Output("p1-dd-graph", "figure"),
    Output("p1-monthly-heatmap", "figure"),
    Output("p1-pnl-histogram", "figure"),
    Input("p1-strategy-checklist", "value"),
    Input("p1-overview-mode", "value"),
)
def update_overview(selected_strategy_ids, mode):
    """
    Build Overview metrics and charts based on the currently selected strategies
    and the display mode (cumulative vs all-selected).
    """
    selected_strategy_ids = selected_strategy_ids or []
    mode = mode or "cumulative"
    

    # Map strategy_id -> short label for legend/hover
    label_map = {}
    for sid in selected_strategy_ids:
        meta = sh_layout.p1_strategy_store.get(sid, {})
        name = meta.get("name")
        if name:
            label_map[sid] = name
        else:
            # Fallback: derive from id if it's a path
            base = os.path.basename(sid)
            if base.lower().endswith(".csv"):
                base = base[:-4]
            label_map[sid] = base or sid


    # Empty figures template
    empty_layout = {
        "template": "plotly_dark",
        "paper_bgcolor": "#222222",
        "plot_bgcolor": "#222222",
        "font": {"color": "#EEEEEE"},
        "xaxis": {"visible": False},
        "yaxis": {"visible": False},
        "annotations": [
            {
                "text": "No data",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"color": "#888888"},
            }
        ],
    }
    empty_fig = {"data": [], "layout": empty_layout}

    # If no strategies selected: show empty metrics and empty figures
    if not selected_strategy_ids:
        metrics_children = html.Div(
            "No strategies selected.",
            style={"color": "#AAAAAA", "fontSize": "0.85rem"},
        )
        return metrics_children, empty_fig, empty_fig, empty_fig, empty_fig

    # ----------------------------------------------------------
    # Collect trades for all selected strategies
    # ----------------------------------------------------------
    frames = []
    for sid in selected_strategy_ids:
        df = _get_strategy_trades(sid)
        if df.empty:
            continue

        # Normalize
        df = df.copy()
        df["Date Closed"] = pd.to_datetime(df["Date Closed"], errors="coerce")
        df = df.dropna(subset=["Date Closed"])
        df["P/L"] = pd.to_numeric(df["P/L"], errors="coerce")
        df = df.dropna(subset=["P/L"])

        if df.empty:
            continue

        df["strategy_id"] = sid
        frames.append(df)

    if not frames:
        metrics_children = html.Div(
            "No valid trades found for selected strategies.",
            style={"color": "#AAAAAA", "fontSize": "0.85rem"},
        )
        return metrics_children, empty_fig, empty_fig, empty_fig, empty_fig

    all_trades = pd.concat(frames, ignore_index=True)

    # ----------------------------------------------------------
    # Per-trade P&L statistics (for metrics)
    # ----------------------------------------------------------
    
    pnl = all_trades["P/L"]
    n_trades = len(pnl)
    total_pnl = pnl.sum()

    # Basic stats
    win_mask = pnl > 0
    lose_mask = pnl < 0
    n_wins = int(win_mask.sum())
    n_losses = int(lose_mask.sum())

    win_rate = n_wins / n_trades if n_trades > 0 else 0.0

    avg_win = pnl[win_mask].mean() if n_wins > 0 else None
    avg_loss = pnl[lose_mask].mean() if n_losses > 0 else None
    if avg_win is not None and avg_loss is not None and avg_loss != 0:
        avg_win_loss_ratio = avg_win / abs(avg_loss)
    else:
        avg_win_loss_ratio = None

    gross_win = pnl[win_mask].sum()
    gross_loss = pnl[lose_mask].sum()
    if gross_loss < 0:
        profit_factor = gross_win / abs(gross_loss)
    else:
        profit_factor = None

    # Sortino per-trade (downside-only volatility, unscaled)
    # Kept for potential future use, but not displayed in Overview.
    downside = pnl[pnl < 0]
    if len(downside) > 0:
        downside_std = downside.std(ddof=1)
        sortino = pnl.mean() / downside_std if downside_std > 0 else None
    else:
        sortino = None

    # Skewness & kurtosis (Fisher)
    skewness = pnl.skew() if n_trades > 2 else None
    kurtosis = pnl.kurtosis() if n_trades > 3 else None

    # Tail ratio – 95th percentile of winners vs 95th of abs(losses)
    pos_tail = pnl[win_mask].quantile(0.95) if n_wins > 0 else None
    neg_tail = pnl[lose_mask].abs().quantile(0.95) if n_losses > 0 else None
    if pos_tail and neg_tail and neg_tail > 0:
        tail_ratio = pos_tail / neg_tail
    else:
        tail_ratio = None

    # % of P&L from top 5 trades (by P&L descending)
    # Stored as a ratio (0–1), we will format as % later.
    if n_trades >= 1 and total_pnl != 0:
        top5_sum = pnl.sort_values(ascending=False).head(5).sum()
        pct_top5 = top5_sum / total_pnl
    else:
        pct_top5 = None

    # % of P&L from top 5% of trades (by P&L descending)
    # Also stored as a ratio (0–1); we will format as % later.
    if n_trades >= 1 and total_pnl != 0:
        top_k = max(1, int(round(0.05 * n_trades)))
        top5pct_sum = pnl.sort_values(ascending=False).head(top_k).sum()
        pct_top5pct = top5pct_sum / total_pnl
    else:
        pct_top5pct = None

    # ----------------------------------------------------------
    # Equity, drawdown, daily returns (aggregated and per-strategy)
    # ----------------------------------------------------------
    # Equity per strategy (cum P&L, starting from 0) – for charts
    per_strat_equity = {}
    for sid in selected_strategy_ids:
        df_sid = all_trades[all_trades["strategy_id"] == sid].copy()
        if df_sid.empty:
            continue
        df_sid = df_sid.sort_values("Date Closed")
        eq = df_sid["P/L"].cumsum()
        per_strat_equity[sid] = pd.Series(eq.values, index=df_sid["Date Closed"].values)

    # Aggregated daily P&L and cumulative P&L (starting at 0)
    daily_pnl = (
        all_trades.groupby("Date Closed")["P/L"].sum().sort_index()
    )
    agg_equity = daily_pnl.cumsum()

    def compute_dd(series: pd.Series):
        if series.empty:
            return series
        cummax = series.cummax()
        return series - cummax

    # Drawdown in dollars for charts (same whether we shift by initial equity or not)
    agg_dd = compute_dd(agg_equity)
    max_dd = agg_dd.min() if not agg_dd.empty else 0.0

    # ----------------------------------------------------------
    # Equity including initial capital, CAGR, DD%, annualised Sharpe
    # ----------------------------------------------------------
    if not agg_equity.empty:
        equity = P1_INITIAL_EQUITY + agg_equity
        equity = equity.sort_index()

        # CAGR based on first/last equity date
        start_date = equity.index[0]
        end_date = equity.index[-1]
        n_days = (end_date - start_date).days
        if n_days > 0 and equity.iloc[-1] > 0:
            years = n_days / DAYS_PER_YEAR
            cagr = (equity.iloc[-1] / P1_INITIAL_EQUITY) ** (1.0 / years) - 1.0
        else:
            cagr = None

        # Max DD in percentage (relative to running peak equity)
        # cummax_eq = equity.cummax()
        # dd_pct_series = (equity - cummax_eq) / cummax_eq.replace(0, np.nan)
        # max_dd_pct = dd_pct_series.min() if not dd_pct_series.empty else None
        
        cummax_eq = equity.cummax()
        dd_pct_series = (equity - cummax_eq) / cummax_eq.replace(0, np.nan)
        
        if not agg_dd.empty:
            trough_date = agg_dd.idxmin()  # trough of max-$ drawdown
            max_dd_pct = dd_pct_series.loc[trough_date]
        else:
            max_dd_pct = None


        # Daily returns & annualised Sharpe (Option A: daily, √252)
        daily_returns = equity.pct_change().dropna()
        if len(daily_returns) >= 2:
            ret_mean = daily_returns.mean()
            ret_std = daily_returns.std(ddof=1)
            if ret_std > 0:
                sharpe_annual = (ret_mean / ret_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
            else:
                sharpe_annual = None
        else:
            sharpe_annual = None
    else:
        cagr = None
        max_dd_pct = None
        sharpe_annual = None

    # MAR in OO style: CAGR / |MaxDD%|
    if cagr is not None and max_dd_pct is not None and max_dd_pct < 0:
        mar_ratio = cagr / abs(max_dd_pct)
    else:
        mar_ratio = None

    # ----------------------------------------------------------
    # Build equity figure (same as before, cum P&L)
    # ----------------------------------------------------------
    equity_traces = []
    
    if mode == "cumulative":
        if not agg_equity.empty:
            equity_traces.append(
                go.Scatter(
                    x=agg_equity.index,
                    y=agg_equity.values,
                    mode="lines",
                    name="Combined",
                )
            )
    else:
        # individual
        for sid, series in per_strat_equity.items():
            if series.empty:
                continue
            
            clean_name = _clean_name(label_map.get(sid, sid))
            
            # Look up color in the shared store (same as weights / page2)
            meta = sh_layout.p1_strategy_store.get(sid, {}) or {}
            color = meta.get("color")
            
            line_kwargs = {"width": 1.5}
            if color:
                line_kwargs["color"] = color
            
            equity_traces.append(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines",
                    name=clean_name,
                    line=line_kwargs,
                    hovertemplate=(
                        f"{clean_name}<br>$%{{y:,.0f}}<br>%{{x|%d-%b-%y}}<extra></extra>"
                    ),
                )
            )

    equity_layout = go.Layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin={"l": 40, "r": 10, "t": 30, "b": 40},
        xaxis={"title": "Date"},
        yaxis={"title": "Equity (cum P&L)"},
        legend={"orientation": "h", "y": -0.2},
    )
    equity_fig = go.Figure(data=equity_traces, layout=equity_layout)

    # ----------------------------------------------------------
    # Build drawdown figure (same as before, in $)
    # ----------------------------------------------------------
    dd_traces = []
    if mode == "cumulative":
        if not agg_dd.empty:
            dd_traces.append(
                go.Scatter(
                    x=agg_dd.index,
                    y=agg_dd.values,
                    mode="lines",
                    name="Combined DD",
                )
            )
    else:
        for sid, series in per_strat_equity.items():
            if series.empty:
                continue
            
            clean_name = _clean_name(label_map.get(sid, sid))
            dd = compute_dd(series)

            meta = sh_layout.p1_strategy_store.get(sid, {}) or {}
            color = meta.get("color")

            line_kwargs = {"width": 1.5}
            if color:
                line_kwargs["color"] = color
            
            dd = compute_dd(series)
            dd_traces.append(
                go.Scatter(
                    x=dd.index,
                    y=dd.values,
                    mode="lines",
                    name=clean_name,
                    line=line_kwargs,
                    hovertemplate=(
                        f"{clean_name}<br>$%{{y:,.0f}}<br>%{{x|%d-%b-%y}}<extra></extra>"
                    ),
                )
            )

    dd_layout = go.Layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin={"l": 40, "r": 10, "t": 30, "b": 40},
        xaxis={"title": "Date"},
        yaxis={"title": "Drawdown"},
        legend={"orientation": "h", "y": -0.2},
    )
    dd_fig = go.Figure(data=dd_traces, layout=dd_layout)

    # ----------------------------------------------------------
    # Monthly returns heatmap (aggregated)
    # ----------------------------------------------------------
    if not daily_pnl.empty:
        monthly = daily_pnl.resample("ME").sum()
        ...


    # ----------------------------------------------------------
    # Monthly returns heatmap (aggregated)
    # ----------------------------------------------------------
    if not daily_pnl.empty:
        monthly = daily_pnl.resample("ME").sum()
        heat_df = monthly.to_frame("P/L")
        heat_df["Year"] = heat_df.index.year
        heat_df["Month"] = heat_df.index.month
        pivot = heat_df.pivot(index="Year", columns="Month", values="P/L")

        # Ensure months 1–12 present as columns
        all_months = list(range(1, 13))
        pivot = pivot.reindex(columns=all_months)

        heat_data = [
            go.Heatmap(
                z=pivot.values,
                x=[datetime(2000, m, 1).strftime("%b") for m in all_months],
                y=pivot.index.astype(str).tolist(),
                colorbar={"title": "P/L"},
                colorscale=[
                    [0.0, "#8b0000"],   # dark red
                    [0.45, "#ff6666"],  # lighter red near zero-
                    [0.50, "#ffffff"],  # thin white center at zero
                    [0.52, "#66ff66"],  # pale green near zero+
                    [1.0, "#009900"],   # dark green
                ],
                zmid=0,
            )
        ]
        heat_layout = go.Layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis={"title": "Month"},
            yaxis={"title": "Year"},
        )
        monthly_fig = go.Figure(data=heat_data, layout=heat_layout)
    else:
        monthly_fig = empty_fig

    # ----------------------------------------------------------
    # P&L distribution histogram (aggregated)
    # ----------------------------------------------------------
    hist_data = [
        go.Histogram(
            x=pnl.values,
            nbinsx=50,
            name="Trade P&L",
            opacity=0.85,
        )
    ]
    hist_layout = go.Layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin={"l": 40, "r": 10, "t": 30, "b": 40},
        xaxis={"title": "Trade P&L"},
        yaxis={"title": "Count"},
        bargap=0.05,
    )
    hist_fig = go.Figure(data=hist_data, layout=hist_layout)


    # ----------------------------------------------------------
    # Metrics panel layout (cards)
    # ----------------------------------------------------------
    def fmt_number(x, decimals=2):
        if x is None:
            return "–"
        try:
            if decimals == 0:
                return f"{x:,.0f}"
            return f"{x:,.{decimals}f}"
        except Exception:
            return "–"

    def fmt_money(x):
        if x is None:
            return "–"
        try:
            return f"${x:,.0f}"
        except Exception:
            return "–"

    def fmt_pct(x, decimals=1):
        if x is None:
            return "–"
        try:
            return f"{100.0 * x:,.{decimals}f}%"
        except Exception:
            return "–"

    header_style = {
        "backgroundColor": "#343a40",
        "borderBottom": "1px solid #444444",
        "fontSize": "0.8rem",
        "textAlign": "center",
        "padding": "0.25rem 0.5rem",
    }
    body_style = {
        "backgroundColor": "#2b2f36",
        "fontSize": "0.9rem",
        "textAlign": "center",
        "padding": "0.4rem 0.5rem",
    }
    card_style = {
        "backgroundColor": "#2b2f36",
        "border": "1px solid #444444",
        "borderRadius": "0.35rem",
    }

    # Metrics cards – two rows, all equal width per row
    top_row = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Total P&L",
                            className="py-1 px-2",
                            style=header_style,
                        ),
                        dbc.CardBody(
                            fmt_money(total_pnl),
                            className="py-1 px-2",
                            style=body_style,
                        ),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Max DD (%)",
                            className="py-1 px-2",
                            style=header_style,
                        ),
                        dbc.CardBody(
                            fmt_pct(max_dd_pct, 2),
                            className="py-1 px-2",
                            style=body_style,
                        ),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "MAR",
                            className="py-1 px-2",
                            style=header_style,
                        ),
                        dbc.CardBody(
                            fmt_number(mar_ratio, 2),
                            className="py-1 px-2",
                            style=body_style,
                        ),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "CAGR",
                            className="py-1 px-2",
                            style=header_style,
                        ),
                        dbc.CardBody(
                            fmt_pct(cagr, 2),
                            className="py-1 px-2",
                            style=body_style,
                        ),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Sharpe (ann.)",
                            className="py-1 px-2",
                            style=header_style,
                        ),
                        dbc.CardBody(
                            fmt_number(sharpe_annual, 2),
                            className="py-1 px-2",
                            style=body_style,
                        ),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Win rate",
                            className="py-1 px-2",
                            style=header_style,
                        ),
                        dbc.CardBody(
                            fmt_pct(win_rate, 1),
                            className="py-1 px-2",
                            style=body_style,
                        ),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Trades",
                            className="py-1 px-2",
                            style=header_style,
                        ),
                        dbc.CardBody(
                            fmt_number(n_trades, 0),
                            className="py-1 px-2",
                            style=body_style,
                        ),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
            ),
        ],
        className="g-2",
    )


    bottom_row = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Avg win / loss",
                            className="py-1 px-2",
                            style=header_style,
                        ),
                        dbc.CardBody(
                            fmt_number(avg_win_loss_ratio, 2),
                            className="py-1 px-2",
                            style=body_style,
                        ),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Profit factor",
                            className="py-1 px-2",
                            style=header_style,
                        ),
                        dbc.CardBody(
                            fmt_number(profit_factor, 2),
                            className="py-1 px-2",
                            style=body_style,
                        ),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Skewness",
                            className="py-1 px-2",
                            style=header_style,
                        ),
                        dbc.CardBody(
                            fmt_number(skewness, 2),
                            className="py-1 px-2",
                            style=body_style,
                        ),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Kurtosis",
                            className="py-1 px-2",
                            style=header_style,
                        ),
                        dbc.CardBody(
                            fmt_number(kurtosis, 2),
                            className="py-1 px-2",
                            style=body_style,
                        ),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Tail ratio",
                            className="py-1 px-2",
                            style=header_style,
                        ),
                        dbc.CardBody(
                            fmt_number(tail_ratio, 2),
                            className="py-1 px-2",
                            style=body_style,
                        ),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "% P&L from top 5",
                            className="py-1 px-2",
                            style=header_style,
                        ),
                        dbc.CardBody(
                            fmt_pct(pct_top5, 1),
                            className="py-1 px-2",
                            style=body_style,
                        ),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "% P&L from top 5%",
                            className="py-1 px-2",
                            style=header_style,
                        ),
                        dbc.CardBody(
                            fmt_pct(pct_top5pct, 1),
                            className="py-1 px-2",
                            style=body_style,
                        ),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
            ),
        ],
        className="g-2",
    )

    metrics_cards = html.Div([top_row, bottom_row])


    return metrics_cards, equity_fig, dd_fig, monthly_fig, hist_fig


@callback(
    Output("p1-entry-manual-container", "style"),
    Input("p1-param-entry-mode-auto", "n_clicks"),
    Input("p1-param-entry-mode-manual", "n_clicks"),
)
def _toggle_entry_time_mode(n_auto, n_manual):
    triggered = ctx.triggered_id
    if triggered == "p1-param-entry-mode-manual":
        # Show sliders when manual is clicked
        return {"display": "block"}
    # Default / auto mode: hide sliders
    return {"display": "none"}

@callback(
    Output("p1-regimes-vix-manual-container", "style"),
    Input("p1-regimes-vix-mode-auto", "n_clicks"),
    Input("p1-regimes-vix-mode-manual", "n_clicks"),
)
def _toggle_vix_mode(n_auto, n_manual):
    triggered = ctx.triggered_id
    if triggered == "p1-regimes-vix-mode-manual":
        # Show sliders when manual is clicked
        return {"display": "block"}
    # Default / auto mode: hide sliders
    return {"display": "none"}


@callback(
    Output("p1-params-margin-bucket", "figure"),
    Output("p1-params-time-bucket", "figure"),
    Output("p1-params-premium-credit", "figure"),
    Output("p1-params-premium-debit", "figure"),
    Output("p1-params-dte", "figure"),
    Output("p1-params-gap", "figure"),
    Output("p1-params-move", "figure"),
    Input("p1-strategy-checklist", "value"),
    Input("p1-param-entry-mode-auto", "n_clicks"),
    Input("p1-param-entry-mode-manual", "n_clicks"),
    Input("p1-entry-slider-1", "value"),
    Input("p1-entry-slider-2", "value"),
    Input("p1-entry-slider-3", "value"),
)
def update_parameters(
    selected_strategy_ids,
    n_auto,
    n_manual,
    s1,
    s2,
    s3,
):

    """
    Parameter sensitivity: how per-trade R multiples vary across:
      - Margin Req. buckets (risk size)
      - Time-of-day buckets (entry time)
      - Premium buckets, split into credit vs debit trades
      - DTE buckets (if DTE column exists)
      - Gap buckets
      - Movement buckets
    """

    # ---------- helper for empty figs ----------
    def _empty_fig(title):
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            title=title,
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    # ---------- load & aggregate trades ----------
    if not selected_strategy_ids:
        empty = _empty_fig("No strategies selected")
        return empty, empty, empty, empty, empty, empty, empty

    frames = []
    for sid in selected_strategy_ids:
        df = _get_strategy_trades(sid)
        if df is None or df.empty:
            continue
        df = df.copy()
        # compute R locally if possible
        if "P/L" in df.columns and "Margin Req." in df.columns:
            pl = pd.to_numeric(df["P/L"], errors="coerce")
            margin = pd.to_numeric(df["Margin Req."], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                df["R"] = pl / margin.replace(0, np.nan)
        frames.append(df)

    if not frames:
        empty = _empty_fig("No trades available")
        return empty, empty, empty, empty, empty, empty, empty

    all_trades = pd.concat(frames, ignore_index=True)

    # ---------- Margin bucket chart ----------
    if "Margin Req." in all_trades.columns and "R" in all_trades.columns:
        df_m = all_trades.dropna(subset=["Margin Req.", "R"]).copy()
        if df_m.empty:
            fig_margin = _empty_fig("Mean R vs margin bucket (no data)")
        else:
            df_m["Margin Req."] = pd.to_numeric(df_m["Margin Req."], errors="coerce")
            df_m = df_m.dropna(subset=["Margin Req."])
            if df_m.empty:
                fig_margin = _empty_fig("Mean R vs margin bucket (no data)")
            else:
                bins = df_m["Margin Req."].quantile([0, 0.25, 0.5, 0.75, 1]).values
                bins = sorted(set(bins))
                if len(bins) < 2:
                    bins = [df_m["Margin Req."].min(), df_m["Margin Req."].max()]
                df_m.loc[:, "bucket"] = pd.cut(
                    df_m["Margin Req."],
                    bins=bins,
                    include_lowest=True,
                    right=True,
                )
                grp = (
                    df_m.dropna(subset=["bucket"])
                    .groupby("bucket", observed=False)["R"]
                    .mean()
                    .reset_index()
                )
                if grp.empty:
                    fig_margin = _empty_fig("Mean R vs margin bucket (no buckets)")
                else:
                    fig_margin = go.Figure(
                        data=[
                            go.Bar(
                                x=grp["bucket"].astype(str),
                                y=grp["R"],
                                text=grp["R"].round(3),
                                textposition="outside",
                            )
                        ]
                    )
                    fig_margin.update_layout(
                        template="plotly_dark",
                        title="Mean R vs margin bucket",
                        xaxis_title="Margin bucket",
                        yaxis_title="Mean R per trade",
                        paper_bgcolor="#222222",
                        plot_bgcolor="#222222",
                    )
    else:
        fig_margin = _empty_fig("Mean R vs margin bucket (no margin/R)")

    # ---------- Entry time bucket chart (AUTO / MANUAL) ----------
    if "Time Opened" in all_trades.columns and "R" in all_trades.columns:
        df_t = all_trades.dropna(subset=["Time Opened", "R"]).copy()
        if df_t.empty:
            fig_time = _empty_fig("Mean R vs entry time bucket (no data)")
        else:
            t_parsed = pd.to_datetime(
                df_t["Time Opened"], format="%H:%M:%S", errors="coerce"
            )
            df_t["t_min"] = t_parsed.dt.hour * 60 + t_parsed.dt.minute
            df_t = df_t.dropna(subset=["t_min"])
            if df_t.empty:
                fig_time = _empty_fig("Mean R vs entry time bucket (no valid times)")
            else:
                # Decide mode
                trig = ctx.triggered_id
                auto_clicks = n_auto or 0
                manual_clicks = n_manual or 0
        
                if trig == "p1-param-entry-mode-manual":
                    manual_mode = True
                elif trig == "p1-param-entry-mode-auto":
                    manual_mode = False
                else:
                    manual_mode = manual_clicks > auto_clicks
        
                if manual_mode:
                    # MANUAL: use slider boundaries (minutes)
                    bounds = [v for v in (s1, s2, s3) if v is not None]
                    bounds = sorted(bounds)
                    # Clamp to [570, 930]
                    bounds = [max(570, min(930, b)) for b in bounds]
        
                    # Final bin edges (including start/end)
                    bins = [570] + bounds + [930]
                    bins = sorted(set(bins))  # remove duplicates, keep ordered
        
                    if len(bins) < 2:
                        fig_time = _empty_fig("Mean R vs entry time bucket (no valid buckets)")
                    else:
                        # Labels as time ranges, e.g. '09:30–10:30'
                        labels = [
                            f"{_minutes_to_hhmm(bins[i])}–{_minutes_to_hhmm(bins[i + 1])}"
                            for i in range(len(bins) - 1)
                        ]
        
                        df_t.loc[:, "bucket"] = pd.cut(
                            df_t["t_min"],
                            bins=bins,
                            labels=labels,
                            include_lowest=True,
                            right=True,
                        )
                else:
                    # AUTO: quantiles on t_min
                    q = df_t["t_min"].quantile([0, 0.25, 0.5, 0.75, 1]).values
                    bins = sorted(set(int(x) for x in q))
        
                    if len(bins) < 2:
                        bins = [df_t["t_min"].min(), df_t["t_min"].max()]
        
                    if len(bins) < 2:
                        fig_time = _empty_fig("Mean R vs entry time bucket (no valid buckets)")
                    else:
                        labels = [
                            f"{_minutes_to_hhmm(bins[i])}–{_minutes_to_hhmm(bins[i + 1])}"
                            for i in range(len(bins) - 1)
                        ]
        
                        df_t.loc[:, "bucket"] = pd.cut(
                            df_t["t_min"],
                            bins=bins,
                            labels=labels,
                            include_lowest=True,
                            right=True,
                        )
        
                # Group & plot (shared for both modes)
                if "bucket" not in df_t.columns:
                    fig_time = _empty_fig("Mean R vs entry time bucket (no buckets)")
                else:
                    grp_t = (
                        df_t.dropna(subset=["bucket"])
                        .groupby("bucket", observed=False)["R"]
                        .mean()
                        .reset_index()
                    )
        
                    if grp_t.empty:
                        fig_time = _empty_fig("Mean R vs entry time bucket (no buckets)")
                    else:
                        fig_time = go.Figure(
                            data=[
                                go.Bar(
                                    x=grp_t["bucket"].astype(str),
                                    y=grp_t["R"],
                                    text=grp_t["R"].round(3),
                                    textposition="outside",
                                )
                            ]
                        )
                        fig_time.update_layout(
                            template="plotly_dark",
                            title="Mean R vs entry time bucket",
                            xaxis_title="Entry time bucket",
                            yaxis_title="Mean R per trade",
                            paper_bgcolor="#222222",
                            plot_bgcolor="#222222",
                        )
                
    else:
        fig_time = _empty_fig("Mean R vs entry time bucket (no time/R)")

    # ---------- Premium credit / debit ----------
    if "Premium" in all_trades.columns and "R" in all_trades.columns:
        df_p = all_trades.dropna(subset=["Premium", "R"]).copy()
        df_p["Premium"] = pd.to_numeric(df_p["Premium"], errors="coerce")
        df_p = df_p.dropna(subset=["Premium"])
        # Credit
        df_c = df_p[df_p["Premium"] > 0].copy()
        if df_c.empty:
            fig_prem_credit = _empty_fig("Mean R vs premium (credit) (no data)")
        else:
            bins_c = df_c["Premium"].quantile([0, 0.25, 0.5, 0.75, 1]).values
            bins_c = sorted(set(bins_c))
            if len(bins_c) < 2:
                bins_c = [df_c["Premium"].min(), df_c["Premium"].max()]
            df_c.loc[:, "bucket"] = pd.cut(
                df_c["Premium"],
                bins=bins_c,
                include_lowest=True,
                right=True,
            )
            grp_c = (
                df_c.dropna(subset=["bucket"])
                .groupby("bucket", observed=False)["R"]
                .mean()
                .reset_index()
            )
            if grp_c.empty:
                fig_prem_credit = _empty_fig("Mean R vs premium (credit) (no buckets)")
            else:
                fig_prem_credit = go.Figure(
                    data=[
                        go.Bar(
                            x=grp_c["bucket"].astype(str),
                            y=grp_c["R"],
                            text=grp_c["R"].round(3),
                            textposition="outside",
                        )
                    ]
                )
                fig_prem_credit.update_layout(
                    template="plotly_dark",
                    title="Mean R vs premium (credit)",
                    xaxis_title="Premium bucket",
                    yaxis_title="Mean R per trade",
                    paper_bgcolor="#222222",
                    plot_bgcolor="#222222",
                )

        # Debit
        df_d = df_p[df_p["Premium"] < 0].copy()
        if df_d.empty:
            fig_prem_debit = _empty_fig("Mean R vs premium (debit) (no data)")
        else:
            bins_d = df_d["Premium"].quantile([0, 0.25, 0.5, 0.75, 1]).values
            bins_d = sorted(set(bins_d))
            if len(bins_d) < 2:
                bins_d = [df_d["Premium"].min(), df_d["Premium"].max()]
            df_d.loc[:, "bucket"] = pd.cut(
                df_d["Premium"],
                bins=bins_d,
                include_lowest=True,
                right=True,
            )
            grp_d = (
                df_d.dropna(subset=["bucket"])
                .groupby("bucket", observed=False)["R"]
                .mean()
                .reset_index()
            )
            if grp_d.empty:
                fig_prem_debit = _empty_fig("Mean R vs premium (debit) (no buckets)")
            else:
                fig_prem_debit = go.Figure(
                    data=[
                        go.Bar(
                            x=grp_d["bucket"].astype(str),
                            y=grp_d["R"],
                            text=grp_d["R"].round(3),
                            textposition="outside",
                        )
                    ]
                )
                fig_prem_debit.update_layout(
                    template="plotly_dark",
                    title="Mean R vs premium (debit)",
                    xaxis_title="Premium bucket",
                    yaxis_title="Mean R per trade",
                    paper_bgcolor="#222222",
                    plot_bgcolor="#222222",
                )
    else:
        fig_prem_credit = _empty_fig("Mean R vs premium (credit) (no premium/R)")
        fig_prem_debit = _empty_fig("Mean R vs premium (debit) (no premium/R)")

    # ---------- DTE bucket chart ----------
    if (
        "Date Opened" in all_trades.columns
        and "Legs" in all_trades.columns
        and "Margin Req." in all_trades.columns
        and "P/L" in all_trades.columns
    ):
        open_dates = pd.to_datetime(all_trades["Date Opened"], errors="coerce")

        MONTHS = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
            "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
            "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
        }

        def _parse_expiry(legs_str, open_dt):
            if pd.isna(open_dt) or not isinstance(legs_str, str):
                return pd.NaT
            tokens = legs_str.replace("|", " ").split()
            month = None
            day = None
            for i, tok in enumerate(tokens):
                if tok in MONTHS and i + 1 < len(tokens):
                    nxt = tokens[i + 1]
                    digits = "".join(ch for ch in nxt if ch.isdigit())
                    if digits:
                        month = MONTHS[tok]
                        day = int(digits)
                        break
            if month is None or day is None:
                return pd.NaT
            year = int(open_dt.year)
            try:
                expiry = datetime(year, month, day)
            except ValueError:
                return pd.NaT
            if expiry.date() < open_dt.date():
                try:
                    expiry = datetime(year + 1, month, day)
                except ValueError:
                    return pd.NaT
            return expiry

        expiry = [
            _parse_expiry(ls, od) for ls, od in zip(all_trades["Legs"], open_dates)
        ]
        expiry = pd.to_datetime(expiry, errors="coerce")
        dte = (expiry - open_dates).dt.days

        pl = pd.to_numeric(all_trades["P/L"], errors="coerce")
        margin = pd.to_numeric(all_trades["Margin Req."], errors="coerce")

        mask = dte.notna() & pl.notna() & margin.notna() & (margin != 0)
        dte = dte[mask]
        r_dte = pl[mask] / margin[mask]

        if dte.empty:
            fig_dte = _empty_fig("Mean R vs DTE bucket (no valid DTE)")
        else:
            dte_bins = [
                (0, 0, "0"),
                (1, 1, "1"),
                (2, 5, "2–5"),
                (6, 10, "6–10"),
                (11, 21, "11–21"),
                (22, 9999, "22+"),
            ]
            labels = []
            vals = []
            for lo, hi, label in dte_bins:
                m = (dte >= lo) & (dte <= hi)
                if m.any():
                    labels.append(label)
                    vals.append(r_dte[m].mean())
            if not labels:
                fig_dte = _empty_fig("Mean R vs DTE bucket (no buckets)")
            else:
                fig_dte = go.Figure(
                    data=[
                        go.Bar(
                            x=labels,
                            y=vals,
                            text=np.round(vals, 3),
                            textposition="outside",
                        )
                    ]
                )
                fig_dte.update_layout(
                    template="plotly_dark",
                    title="Mean R vs DTE bucket",
                    xaxis_title="DTE (days)",
                    yaxis_title="Mean R per trade",
                    paper_bgcolor="#222222",
                    plot_bgcolor="#222222",
                )
    else:
        fig_dte = _empty_fig("Mean R vs DTE bucket (no DTE data)")

    # ---------- Gap bucket ----------
    if "Gap" in all_trades.columns and "R" in all_trades.columns:
        df_g = all_trades.dropna(subset=["Gap", "R"]).copy()
        df_g["Gap"] = pd.to_numeric(df_g["Gap"], errors="coerce")
        df_g = df_g.dropna(subset=["Gap"])
        if df_g.empty:
            fig_gap = _empty_fig("Mean R vs gap bucket (no data)")
        else:
            bins_g = df_g["Gap"].quantile([0, 0.25, 0.5, 0.75, 1]).values
            bins_g = sorted(set(bins_g))
            if len(bins_g) < 2:
                bins_g = [df_g["Gap"].min(), df_g["Gap"].max()]
            df_g.loc[:, "bucket"] = pd.cut(
                df_g["Gap"],
                bins=bins_g,
                include_lowest=True,
                right=True,
            )
            grp_g = (
                df_g.dropna(subset=["bucket"])
                .groupby("bucket", observed=False)["R"]
                .mean()
                .reset_index()
            )
            if grp_g.empty:
                fig_gap = _empty_fig("Mean R vs gap bucket (no buckets)")
            else:
                fig_gap = go.Figure(
                    data=[
                        go.Bar(
                            x=grp_g["bucket"].astype(str),
                            y=grp_g["R"],
                            text=grp_g["R"].round(3),
                            textposition="outside",
                        )
                    ]
                )
                fig_gap.update_layout(
                    template="plotly_dark",
                    title="Mean R vs gap bucket",
                    xaxis_title="Gap bucket",
                    yaxis_title="Mean R per trade",
                    paper_bgcolor="#222222",
                    plot_bgcolor="#222222",
                )
    else:
        fig_gap = _empty_fig("Mean R vs gap bucket (no gap/R)")

    # ---------- Movement bucket ----------
    if "Movement" in all_trades.columns and "R" in all_trades.columns:
        df_mv = all_trades.dropna(subset=["Movement", "R"]).copy()
        df_mv["Movement"] = pd.to_numeric(df_mv["Movement"], errors="coerce")
        df_mv = df_mv.dropna(subset=["Movement"])
        if df_mv.empty:
            fig_move = _empty_fig("Mean R vs movement bucket (no data)")
        else:
            bins_mv = df_mv["Movement"].quantile([0, 0.25, 0.5, 0.75, 1]).values
            bins_mv = sorted(set(bins_mv))
            if len(bins_mv) < 2:
                bins_mv = [df_mv["Movement"].min(), df_mv["Movement"].max()]
            df_mv.loc[:, "bucket"] = pd.cut(
                df_mv["Movement"],
                bins=bins_mv,
                include_lowest=True,
                right=True,
            )
            grp_mv = (
                df_mv.dropna(subset=["bucket"])
                .groupby("bucket", observed=False)["R"]
                .mean()
                .reset_index()
            )
            if grp_mv.empty:
                fig_move = _empty_fig("Mean R vs movement bucket (no buckets)")
            else:
                fig_move = go.Figure(
                    data=[
                        go.Bar(
                            x=grp_mv["bucket"].astype(str),
                            y=grp_mv["R"],
                            text=grp_mv["R"].round(3),
                            textposition="outside",
                        )
                    ]
                )
                fig_move.update_layout(
                    template="plotly_dark",
                    title="Mean R vs movement bucket",
                    xaxis_title="Movement bucket",
                    yaxis_title="Mean R per trade",
                    paper_bgcolor="#222222",
                    plot_bgcolor="#222222",
                )
    else:
        fig_move = _empty_fig("Mean R vs movement bucket (no move/R)")

    return (
        fig_margin,
        fig_time,
        fig_prem_credit,
        fig_prem_debit,
        fig_dte,
        fig_gap,
        fig_move,
    )


@callback(
    Output("p1-regimes-vix-box", "figure"),
    Output("p1-regimes-vix-bar", "figure"),
    Output("p1-regimes-trend-box", "figure"),
    Output("p1-regimes-trend-bar", "figure"),
    Output("p1-regimes-gap-box", "figure"),
    Output("p1-regimes-gap-bar", "figure"),
    Input("p1-strategy-checklist", "value"),
    Input("p1-regimes-vix-mode-auto", "n_clicks"),
    Input("p1-regimes-vix-mode-manual", "n_clicks"),
    Input("p1-regimes-vix-slider-1", "value"),
    Input("p1-regimes-vix-slider-2", "value"),
    Input("p1-regimes-vix-slider-3", "value"),
)
def update_p1_regimes(
    selected_strategy_ids,
    n_auto,
    n_manual,
    v1,
    v2,
    v3,
):

    """
    Regime analysis for Phase 1.

    Uses per-trade R multiples and simple regime labels based on:
      - Opening VIX (quartile-based buckets)
      - VIX trend while the trade is open (rising / flat / falling)
      - Gap in points (direction + magnitude)
    """

    def _empty_fig(title):
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            title={"text": title, "x": 0.01, "xanchor": "left"},
        )
        return fig

    empty_vix_box = _empty_fig("No data for Opening VIX regimes")
    empty_vix_bar = _empty_fig("No data for Opening VIX regimes")
    empty_trend_box = _empty_fig("No data for VIX trend regimes")
    empty_trend_bar = _empty_fig("No data for VIX trend regimes")
    empty_gap_box = _empty_fig("No data for Gap regimes")
    empty_gap_bar = _empty_fig("No data for Gap regimes")

    if not selected_strategy_ids:
        return (
            empty_vix_box,
            empty_vix_bar,
            empty_trend_box,
            empty_trend_bar,
            empty_gap_box,
            empty_gap_bar,
        )

    # -----------------------
    # Collect trades and R
    # -----------------------
    frames = []
    for sid in selected_strategy_ids:
        df = _get_strategy_trades(sid)
        if df is None or df.empty:
            continue
        df = df.copy()
        # Compute R where possible
        if "P/L" in df.columns and "Margin Req." in df.columns:
            pl = pd.to_numeric(df["P/L"], errors="coerce")
            margin = pd.to_numeric(df["Margin Req."], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                df["R"] = pl / margin.replace(0, np.nan)
        frames.append(df)

    if not frames:
        return (
            empty_vix_box,
            empty_vix_bar,
            empty_trend_box,
            empty_trend_bar,
            empty_gap_box,
            empty_gap_bar,
        )

    all_trades = pd.concat(frames, ignore_index=True)

    # We need R for all regime charts
    if "R" not in all_trades.columns:
        return (
            empty_vix_box,
            empty_vix_bar,
            empty_trend_box,
            empty_trend_bar,
            empty_gap_box,
            empty_gap_bar,
        )

    # -----------------------
    # VIX level at entry (Opening VIX quartiles)
    # -----------------------
    if "Opening VIX" in all_trades.columns:
        df_v = all_trades.dropna(subset=["Opening VIX", "R"]).copy()
        df_v["Opening VIX"] = pd.to_numeric(df_v["Opening VIX"], errors="coerce")
        df_v = df_v.dropna(subset=["Opening VIX", "R"])
    else:
        df_v = pd.DataFrame(columns=["Opening VIX", "R"])
    
    if not df_v.empty:
        vix_vals = df_v["Opening VIX"]
    
        # Decide mode (AUTO vs MANUAL) using same pattern as entry-time section
        trig = ctx.triggered_id
        auto_clicks = n_auto or 0
        manual_clicks = n_manual or 0
    
        if trig == "p1-regimes-vix-mode-manual":
            manual_mode = True
        elif trig == "p1-regimes-vix-mode-auto":
            manual_mode = False
        else:
            manual_mode = manual_clicks > auto_clicks
    
        if manual_mode:
            # MANUAL: use slider boundaries
            bounds = [v for v in (v1, v2, v3) if v is not None]
            bounds = sorted(bounds)
            # Clamp to slider range
            bounds = [
                max(VIX_SLIDER_MIN, min(VIX_SLIDER_MAX, float(b))) for b in bounds
            ]
            # Build bins from global min/max and sliders
            bins = [VIX_SLIDER_MIN] + bounds + [VIX_SLIDER_MAX]
            # Remove duplicates and sort to avoid ValueError in pd.cut
            bins = sorted(set(bins))
        else:
            # AUTO: quartiles on Opening VIX
            bins = vix_vals.quantile([0, 0.25, 0.5, 0.75, 1]).values.tolist()
            bins = sorted(set(float(x) for x in bins))
    
        # Fallbacks if bins degenerate
        if len(bins) < 2:
            bins = [float(vix_vals.min()), float(vix_vals.max())]
    
        if bins[0] == bins[-1]:
            # All values identical; no meaningful regimes
            df_v["vix_bucket"] = pd.IntervalIndex.from_arrays(
                [bins[0]],
                [bins[-1]],
                closed="both",
            )[0]
        else:
            df_v["vix_bucket"] = pd.cut(
                df_v["Opening VIX"],
                bins=bins,
                include_lowest=True,
                right=True,
            )
        df_v = df_v.dropna(subset=["vix_bucket"])


    if df_v.empty:
        fig_vix_box = empty_vix_box
        fig_vix_bar = empty_vix_bar
    else:
        vix_categories = df_v["vix_bucket"].cat.categories

        # Boxplot: R by Opening VIX bucket
        fig_vix_box = go.Figure()
        for cat in vix_categories:
            vals = df_v.loc[df_v["vix_bucket"] == cat, "R"]
            if vals.empty:
                continue
            fig_vix_box.add_trace(
                go.Box(
                    x=[str(cat)] * len(vals),
                    y=vals,
                    name=str(cat),
                    boxmean="sd",
                    showlegend=False,
                )
            )
        fig_vix_box.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            title={
                "text": "R distribution by Opening VIX regime",
                "x": 0.01,
                "xanchor": "left",
            },
            xaxis={"title": "Opening VIX quartile bucket"},
            yaxis={"title": "R per trade"},
        )

        # Bar chart: mean R by Opening VIX bucket
        grp_vix = (
            df_v.groupby("vix_bucket", observed=False)["R"]
            .agg(["mean", "count"])
            .reset_index()
        )
        grp_vix["bucket_str"] = grp_vix["vix_bucket"].astype(str)
        fig_vix_bar = go.Figure(
            data=[
                go.Bar(
                    x=grp_vix["bucket_str"],
                    y=grp_vix["mean"],
                    text=[f"N={c}" for c in grp_vix["count"]],
                    textposition="outside",
                )
            ]
        )
        fig_vix_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            title={
                "text": "Mean R by Opening VIX regime",
                "x": 0.01,
                "xanchor": "left",
            },
            xaxis={"title": "Opening VIX quartile bucket"},
            yaxis={"title": "Mean R per trade"},
            uniformtext_minsize=8,
            uniformtext_mode="hide",
        )

    # -----------------------
    # VIX trend while in trade (exclude 0-DTE)
    # -----------------------
    if (
        "Opening VIX" in all_trades.columns
        and "Closing VIX" in all_trades.columns
        and "Date Opened" in all_trades.columns
        and "Date Closed" in all_trades.columns
    ):
        df_t = all_trades.dropna(
            subset=["Opening VIX", "Closing VIX", "Date Opened", "Date Closed", "R"]
        ).copy()
        if not df_t.empty:
            df_t["Opening VIX"] = pd.to_numeric(df_t["Opening VIX"], errors="coerce")
            df_t["Closing VIX"] = pd.to_numeric(df_t["Closing VIX"], errors="coerce")
            df_t["Date Opened"] = pd.to_datetime(df_t["Date Opened"], errors="coerce")
            df_t["Date Closed"] = pd.to_datetime(df_t["Date Closed"], errors="coerce")
            df_t = df_t.dropna(
                subset=["Opening VIX", "Closing VIX", "Date Opened", "Date Closed", "R"]
            )
        else:
            df_t = pd.DataFrame(columns=["Opening VIX", "Closing VIX", "Date Opened", "Date Closed", "R"])
    else:
        df_t = pd.DataFrame(columns=["Opening VIX", "Closing VIX", "Date Opened", "Date Closed", "R"])

    if not df_t.empty:
        duration = (df_t["Date Closed"] - df_t["Date Opened"]).dt.days
        df_t = df_t[duration > 0].copy()  # exclude 0-DTE trades
        if not df_t.empty:
            df_t["delta_vix"] = df_t["Closing VIX"] - df_t["Opening VIX"]
            df_t["vix_trend"] = np.where(
                df_t["delta_vix"] > 0,
                "VIX rising",
                np.where(
                    df_t["delta_vix"] < 0,
                    "VIX falling",
                    "VIX flat",
                ),
            )
            df_t = df_t.dropna(subset=["vix_trend", "R"])
        else:
            df_t = pd.DataFrame(columns=["vix_trend", "R"])

    if df_t.empty:
        fig_trend_box = empty_trend_box
        fig_trend_bar = empty_trend_bar
    else:
        trend_order = ["VIX falling", "VIX flat", "VIX rising"]
        df_t["vix_trend"] = pd.Categorical(
            df_t["vix_trend"],
            categories=trend_order,
            ordered=True,
        )

        # Boxplot: R by VIX trend
        fig_trend_box = go.Figure()
        for cat in trend_order:
            vals = df_t.loc[df_t["vix_trend"] == cat, "R"]
            if vals.empty:
                continue
            fig_trend_box.add_trace(
                go.Box(
                    x=[cat] * len(vals),
                    y=vals,
                    name=cat,
                    boxmean="sd",
                    showlegend=False,
                )
            )
        fig_trend_box.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            title={
                "text": "R distribution by VIX trend while in trade",
                "x": 0.01,
                "xanchor": "left",
            },
            xaxis={"title": "VIX trend"},
            yaxis={"title": "R per trade"},
        )

        # Bar chart: mean R by VIX trend
        grp_trend = (
            df_t.groupby("vix_trend", observed=False)["R"]
            .agg(["mean", "count"])
            .reset_index()
        )
        fig_trend_bar = go.Figure(
            data=[
                go.Bar(
                    x=grp_trend["vix_trend"],
                    y=grp_trend["mean"],
                    text=[f"N={c}" for c in grp_trend["count"]],
                    textposition="outside",
                )
            ]
        )
        fig_trend_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            title={
                "text": "Mean R by VIX trend while in trade",
                "x": 0.01,
                "xanchor": "left",
            },
            xaxis={"title": "VIX trend"},
            yaxis={"title": "Mean R per trade"},
            uniformtext_minsize=8,
            uniformtext_mode="hide",
        )

    # -----------------------
    # Gap regimes (direction + magnitude)
    # -----------------------
    if "Gap" in all_trades.columns:
        df_g = all_trades.dropna(subset=["Gap", "R"]).copy()
        df_g["Gap"] = pd.to_numeric(df_g["Gap"], errors="coerce")
        df_g = df_g.dropna(subset=["Gap", "R"])
    else:
        df_g = pd.DataFrame(columns=["Gap", "R"])

    if not df_g.empty:
        abs_gap = df_g["Gap"].abs()
        # 75th percentile threshold for "large" gaps
        T = abs_gap.quantile(0.75)
        if T <= 0:
            # Fallback: use max as threshold if quantile is zero or negative
            T = abs_gap.max()
        if T <= 0:
            # Still degenerate: no variation, so no meaningful regimes
            df_g = pd.DataFrame(columns=["gap_bucket", "R"])
        else:
            conditions = [
                df_g["Gap"] <= -T,
                (df_g["Gap"] < 0) & (df_g["Gap"] > -T),
                (df_g["Gap"] >= 0) & (df_g["Gap"] < T),
                df_g["Gap"] >= T,
            ]
            choices = [
                "Large gap down",
                "Small gap down",
                "Small gap up",
                "Large gap up",
            ]
            df_g["gap_bucket"] = np.select(conditions, choices, default=np.nan)
            df_g = df_g.dropna(subset=["gap_bucket", "R"])
    else:
        df_g = pd.DataFrame(columns=["gap_bucket", "R"])

    if df_g.empty:
        fig_gap_box = empty_gap_box
        fig_gap_bar = empty_gap_bar
    else:
        gap_order = [
            "Large gap down",
            "Small gap down",
            "Small gap up",
            "Large gap up",
        ]
        df_g["gap_bucket"] = pd.Categorical(
            df_g["gap_bucket"],
            categories=gap_order,
            ordered=True,
        )

        # Boxplot: R by Gap regime
        fig_gap_box = go.Figure()
        for cat in gap_order:
            vals = df_g.loc[df_g["gap_bucket"] == cat, "R"]
            if vals.empty:
                continue
            fig_gap_box.add_trace(
                go.Box(
                    x=[cat] * len(vals),
                    y=vals,
                    name=cat,
                    boxmean="sd",
                    showlegend=False,
                )
            )
        fig_gap_box.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            title={
                "text": "R distribution by Gap regime",
                "x": 0.01,
                "xanchor": "left",
            },
            xaxis={"title": "Gap regime"},
            yaxis={"title": "R per trade"},
        )

        # Bar chart: mean R by Gap regime
        grp_gap = (
            df_g.groupby("gap_bucket", observed=False)["R"]
            .agg(["mean", "count"])
            .reset_index()
        )
        fig_gap_bar = go.Figure(
            data=[
                go.Bar(
                    x=grp_gap["gap_bucket"],
                    y=grp_gap["mean"],
                    text=[f"N={c}" for c in grp_gap["count"]],
                    textposition="outside",
                )
            ]
        )
        fig_gap_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            title={
                "text": "Mean R by Gap regime",
                "x": 0.01,
                "xanchor": "left",
            },
            xaxis={"title": "Gap regime"},
            yaxis={"title": "Mean R per trade"},
            uniformtext_minsize=8,
            uniformtext_mode="hide",
        )

    return (
        fig_vix_box,
        fig_vix_bar,
        fig_trend_box,
        fig_trend_bar,
        fig_gap_box,
        fig_gap_bar,
    )


@callback(
    Output("p1-tail-metrics", "children"),
    Output("p1-tail-hist", "figure"),
    Output("p1-tail-lossconcentration", "figure"),
    Output("p1-tail-ddoverlay", "figure"),
    Output("p1-tail-conditional", "figure"),
    Input("p1-strategy-checklist", "value"),
    Input("p1-tail-horizon", "value"),
)
def update_tail_risk(selected_strategy_ids, horizon):
    """
    Tail risk & fragility diagnostics, per-trade, for the currently selected strategies.
    """

    selected_strategy_ids = selected_strategy_ids or []
    horizon = horizon or 10

    # Helpers (same style as Overview)
    def fmt_number(x, decimals=2):
        if x is None:
            return "–"
        try:
            if decimals == 0:
                return f"{x:,.0f}"
            return f"{x:,.{decimals}f}"
        except Exception:
            return "–"

    def fmt_money(x):
        if x is None:
            return "–"
        try:
            return f"${x:,.0f}"
        except Exception:
            return "–"

    def fmt_pct(x, decimals=1):
        if x is None:
            return "–"
        try:
            return f"{100.0 * x:,.{decimals}f}%"
        except Exception:
            return "–"

    header_style = {
        "backgroundColor": "#343a40",
        "borderBottom": "1px solid #444444",
        "fontSize": "0.8rem",
        "textAlign": "center",
        "padding": "0.25rem 0.5rem",
    }
    body_style = {
        "backgroundColor": "#2b2f36",
        "fontSize": "0.9rem",
        "textAlign": "center",
        "padding": "0.4rem 0.5rem",
    }
    card_style = {
        "backgroundColor": "#2b2f36",
        "border": "1px solid #444444",
        "borderRadius": "0.35rem",
    }

    def make_empty_fig(message):
        layout = go.Layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": message,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"color": "#AAAAAA"},
                }
            ],
        )
        return go.Figure(data=[], layout=layout)

    # If no strategies selected -> empty
    if not selected_strategy_ids:
        empty_cards = html.Div(
            "Select at least one strategy to see tail diagnostics.",
            style={"color": "#AAAAAA", "fontSize": "0.85rem"},
        )
        empty_fig = make_empty_fig("No data")
        return empty_cards, empty_fig, empty_fig, empty_fig, empty_fig

    # Aggregate trades from all selected strategies
    frames = []
    for sid in selected_strategy_ids:
        df = _get_strategy_trades(sid)
        if df is None or df.empty:
            continue
        if "P/L" not in df.columns:
            continue
        tmp = df[["Date Closed", "P/L"]].copy()
        frames.append(tmp)

    if not frames:
        empty_cards = html.Div(
            "No valid trade data for the selected strategies.",
            style={"color": "#AAAAAA", "fontSize": "0.85rem"},
        )
        empty_fig = make_empty_fig("No data")
        return empty_cards, empty_fig, empty_fig, empty_fig, empty_fig

    trades = pd.concat(frames, ignore_index=True)
    trades["P/L"] = pd.to_numeric(trades["P/L"], errors="coerce")
    trades = trades.dropna(subset=["P/L"])

    trades["Date Closed"] = pd.to_datetime(trades["Date Closed"], errors="coerce")
    trades = trades.dropna(subset=["Date Closed"])

    if trades.empty:
        empty_cards = html.Div(
            "No usable trades after cleaning.",
            style={"color": "#AAAAAA", "fontSize": "0.85rem"},
        )
        empty_fig = make_empty_fig("No data")
        return empty_cards, empty_fig, empty_fig, empty_fig, empty_fig

    # Sort chronologically
    trades = trades.sort_values("Date Closed").reset_index(drop=True)
    pnl = trades["P/L"]

    n_trades = len(pnl)

    # --- Tail thresholds and ETL ---
    q05 = pnl.quantile(0.05) if n_trades > 0 else None
    q01 = pnl.quantile(0.01) if n_trades > 0 else None

    if q05 is not None:
        etl5 = pnl[pnl <= q05].mean()
        n_tail = (pnl <= q05).sum()
        tail_freq = float(n_tail) / float(n_trades) if n_trades > 0 else None
    else:
        etl5 = None
        tail_freq = None

    etl1 = pnl[pnl <= q01].mean() if q01 is not None else None

    worst_loss = pnl.min() if n_trades > 0 else None

    # --- Max monthly loss ---
    monthly = trades.set_index("Date Closed")["P/L"].resample("ME").sum()
    max_monthly_loss = monthly.min() if not monthly.empty else None

    # --- Worst / average loss ratios ---
    losers = pnl[pnl < 0]
    if not losers.empty:
        avg_loss = losers.mean()  # negative
        if avg_loss != 0:
            worst_over_avg = abs(worst_loss) / abs(avg_loss) if worst_loss is not None else None
        else:
            worst_over_avg = None

        sorted_pnl = pnl.sort_values()  # ascending: most negative first
        if len(sorted_pnl) >= 2:
            second_worst = sorted_pnl.iloc[1]
            if avg_loss != 0:
                second_over_avg = abs(second_worst) / abs(avg_loss)
            else:
                second_over_avg = None
        else:
            second_over_avg = None
    else:
        worst_over_avg = None
        second_over_avg = None

    # --- Median trades between tail events (5%) ---
    if q05 is not None:
        is_tail = pnl <= q05
        tail_positions = trades.index[is_tail]
        if len(tail_positions) >= 2:
            gaps = tail_positions.to_series().diff().dropna()
            median_gap = float(gaps.median())
        else:
            median_gap = None
    else:
        median_gap = None

    # --- Metrics cards layout ---
    metrics_cards = dbc.Row(
        [
            # Row 1
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("ETL 5%", className="py-1 px-2", style=header_style),
                        dbc.CardBody(fmt_money(etl5), className="py-1 px-2", style=body_style),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("ETL 1%", className="py-1 px-2", style=header_style),
                        dbc.CardBody(fmt_money(etl1), className="py-1 px-2", style=body_style),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Max trade loss", className="py-1 px-2", style=header_style),
                        dbc.CardBody(fmt_money(worst_loss), className="py-1 px-2", style=body_style),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Max monthly loss", className="py-1 px-2", style=header_style),
                        dbc.CardBody(fmt_money(max_monthly_loss), className="py-1 px-2", style=body_style),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
                md=3,
            ),

            # Row 2
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Tail event frequency (5%)", className="py-1 px-2", style=header_style),
                        dbc.CardBody(fmt_pct(tail_freq, 2), className="py-1 px-2", style=body_style),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Worst loss / avg loss", className="py-1 px-2", style=header_style),
                        dbc.CardBody(fmt_number(worst_over_avg, 2), className="py-1 px-2", style=body_style),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("2nd worst / avg loss", className="py-1 px-2", style=header_style),
                        dbc.CardBody(fmt_number(second_over_avg, 2), className="py-1 px-2", style=body_style),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Median trades between tail events", className="py-1 px-2", style=header_style),
                        dbc.CardBody(fmt_number(median_gap, 0), className="py-1 px-2", style=body_style),
                    ],
                    className="mb-2",
                    style=card_style,
                ),
                md=3,
            ),
        ],
        className="g-2",
    )

    # === Chart A: log-scale P&L histogram ===
    hist_data = [
        go.Histogram(
            x=pnl.values,
            nbinsx=50,
            name="Trade P&L",
            opacity=0.85,
        )
    ]
    hist_layout = go.Layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin={"l": 40, "r": 10, "t": 30, "b": 40},
        xaxis={"title": "Trade P&L"},
        yaxis={"title": "Count", "type": "log"},
        bargap=0.05,
    )
    hist_fig = go.Figure(data=hist_data, layout=hist_layout)

    shapes = []
    if q05 is not None:
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": float(q05),
                "x1": float(q05),
                "y0": 0,
                "y1": 1,
                "line": {"color": "#ff7f0e", "width": 2, "dash": "dash"},
            }
        )
    if q01 is not None:
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": float(q01),
                "x1": float(q01),
                "y0": 0,
                "y1": 1,
                "line": {"color": "#d62728", "width": 2, "dash": "dash"},
            }
        )
    if shapes:
        hist_fig.update_layout(shapes=shapes)

    # === Chart B: loss concentration (remove N worst losses) ===
    if n_trades > 0:
        sorted_pnl = pnl.sort_values()  # ascending
        ns = [0, 1, 2, 3, 5, 10]
        ns = [n for n in ns if n < n_trades]  # cannot remove >= all trades
        x_vals = []
        y_vals = []
        for n in ns:
            trimmed = sorted_pnl.iloc[n:]
            x_vals.append(n)
            y_vals.append(trimmed.sum())

        conc_data = [
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name="Total P&L excl. N worst",
            )
        ]
        conc_layout = go.Layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis={"title": "Number of worst losses removed"},
            yaxis={"title": "Total P&L ($)"},
        )
        conc_fig = go.Figure(data=conc_data, layout=conc_layout)
    else:
        conc_fig = make_empty_fig("No data")

        # === Chart C: DD overlay of worst 3 periods (per trade index) ===
    eq = pnl.cumsum()
    peak = eq.cummax()
    dd = eq - peak  # negative or zero

    if len(dd) > 0:
        # find indices of worst 3 drawdown troughs
        worst_dd = dd.nsmallest(min(3, len(dd)))
        indices = worst_dd.index.tolist()
        window = 20  # trades before/after trough

        dd_traces = []
        for idx in indices:
            # slice up to 20 trades before and after the trough
            start = max(0, idx - window)
            end = min(len(dd) - 1, idx + window)

            segment = dd.iloc[start : end + 1]

            # shift so that the trough itself (at idx) is at 0 on the Y axis
            trough_value = dd.iloc[idx]
            segment_shifted = segment - trough_value

            # x-axis: 0 = trough, negatives = trades before, positives = trades after
            trades_before = idx - start
            x_seg = list(range(-trades_before, -trades_before + len(segment_shifted)))

            dd_traces.append(
                go.Scatter(
                    x=x_seg,
                    y=segment_shifted.values,
                    mode="lines",
                    name=f"DD around trade {idx}",
                )
            )

        dd_layout = go.Layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis={"title": "Trades relative to trough (0 = worst point)"},
            yaxis={"title": "Drawdown from local peak ($, shifted at trough)"},
            legend={"orientation": "h", "y": -0.2},
        )
        dd_fig = go.Figure(data=dd_traces, layout=dd_layout)
    else:
        dd_fig = make_empty_fig("No data")

    # === Chart D: large-loss conditional return (next N trades) ===
    H = int(horizon) if horizon else 10
    if q05 is not None and H > 0:
        is_tail = pnl <= q05
        tail_indices = trades.index[is_tail]
        path_traces = []
        all_paths = []

        for idx in tail_indices:
            # subsequent trades after the tail event
            subsequent = pnl.iloc[idx + 1 : idx + 1 + H]
            if subsequent.empty:
                continue
            cum_path = subsequent.cumsum()
            x_path = list(range(1, len(cum_path) + 1))

            path_traces.append(
                go.Scatter(
                    x=x_path,
                    y=cum_path.values,
                    mode="lines",
                    line={"width": 1},
                    opacity=0.25,
                    showlegend=False,
                )
            )

            # pad to horizon length for averaging
            path_full = cum_path.reindex(range(1, H + 1)).ffill().fillna(0.0)
            all_paths.append(path_full.values)

        cond_data = path_traces
        if all_paths:
            paths_df = pd.DataFrame(all_paths).T  # shape: H x n_events
            avg_path = paths_df.mean(axis=1)
            x_avg = list(range(1, len(avg_path) + 1))
            cond_data.append(
                go.Scatter(
                    x=x_avg,
                    y=avg_path.values,
                    mode="lines",
                    line={"width": 3},
                    name="Average after tail",
                )
            )

            cond_layout = go.Layout(
                template="plotly_dark",
                paper_bgcolor="#222222",
                plot_bgcolor="#222222",
                font={"color": "#EEEEEE"},
                margin={"l": 40, "r": 10, "t": 30, "b": 40},
                xaxis={"title": f"Trades after tail event (N={H})"},
                yaxis={"title": "Cumulative P&L after tail ($)"},
                showlegend=True,
            )
            cond_fig = go.Figure(data=cond_data, layout=cond_layout)
        else:
            cond_fig = make_empty_fig("No tail events with enough subsequent trades")
    else:
        cond_fig = make_empty_fig("No tail events or horizon invalid")

    return metrics_cards, hist_fig, conc_fig, dd_fig, cond_fig


# === Overfitting helpers =====================================================

OF_MAX_FAN_PATHS = 100  # max number of MC paths to plot in the fan chart


def _of_prepare_series(selected_strategy_ids, metric_mode):
    """
    Collect trades across selected strategies and return a dict with:
        series       : 1D np.array of metric values (R or P&L)
        metric_label : "R-multiple" or "P&L ($)"
        unit_label   : "R" or "$"
        n_trades     : length of series
        equity_actual: cumulative sum of series
    """
    if not selected_strategy_ids:
        return None

    frames = []
    for sid in selected_strategy_ids:
        df = _get_strategy_trades(sid)
        if df is None or df.empty:
            continue
        df = df.copy()

        # Ensure numeric P/L
        if "P/L" in df.columns:
            df["P/L"] = pd.to_numeric(df["P/L"], errors="coerce")
        else:
            df["P/L"] = np.nan

        # Compute R if Margin Req. available
        if "Margin Req." in df.columns:
            margin = pd.to_numeric(df["Margin Req."], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                df["R"] = df["P/L"] / margin.replace(0, np.nan)
        else:
            df["R"] = np.nan

        frames.append(df)

    if not frames:
        return None

    all_trades = pd.concat(frames, ignore_index=True)
    
    # NEW: sort trades chronologically so "actual" metrics match Overview behaviour
    if "Date Closed" in all_trades.columns:
        all_trades["Date Closed"] = pd.to_datetime(all_trades["Date Closed"])
        all_trades = all_trades.sort_values("Date Closed").reset_index(drop=True)

    metric_mode = metric_mode or "R"
    metric_mode = "PL" if metric_mode == "PL" else "R"

    if metric_mode == "PL":
        series = pd.to_numeric(all_trades["P/L"], errors="coerce").dropna()
        metric_label = "P&L ($)"
        unit_label = "$"
    else:
        series = pd.to_numeric(all_trades["R"], errors="coerce").dropna()
        metric_label = "R-multiple"
        unit_label = "R"

    vals = series.values.astype(float)
    n_trades = len(vals)
    equity_actual = np.cumsum(vals) if n_trades > 0 else np.array([])

    return {
        "series": vals,
        "metric_label": metric_label,
        "unit_label": unit_label,
        "n_trades": n_trades,
        "equity_actual": equity_actual,
    }


def _of_compute_max_drawdown_from_series(series):
    """Max drawdown on cumulative sum of a 1D series (returns R or P&L)."""
    vals = np.asarray(series, dtype=float)
    if vals.size == 0:
        return 0.0
    equity = np.cumsum(vals)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity - running_max
    return float(drawdowns.min())  # negative or 0


def _of_compute_losing_streak_stats(series):
    """
    Return (max_consecutive_losses, avg_consecutive_losses_len_ge_2)
    for a 1D series (loss = value < 0).
    """
    vals = np.asarray(series, dtype=float)
    if vals.size == 0:
        return 0, 0.0

    losses = vals < 0
    streaks = []
    cur = 0
    for is_loss in losses:
        if is_loss:
            cur += 1
        elif cur > 0:
            streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)

    if not streaks:
        return 0, 0.0

    max_streak = int(max(streaks))
    long_streaks = [s for s in streaks if s >= 2]
    avg_long = float(np.mean(long_streaks)) if long_streaks else 0.0
    return max_streak, avg_long


def _of_bootstrap_sharpes(vals, n_boot=500, block_len=1):
    """
    Block bootstrap of Sharpe ratios for a 1D array of returns (R or P&L).
    block_len = 1 => standard iid bootstrap.
    """
    vals = np.asarray(vals, dtype=float)
    N = vals.size
    if N < 2 or n_boot is None or n_boot <= 0:
        return np.array([])

    block_len = int(block_len) if block_len is not None else 1
    if block_len < 1:
        block_len = 1
    if block_len > N:
        block_len = N

    sharpes = []

    if block_len == 1:
        # Standard iid bootstrap
        for _ in range(int(n_boot)):
            sample = np.random.choice(vals, size=N, replace=True)
            m = sample.mean()
            s = sample.std(ddof=1)
            if s > 0:
                sharpes.append(m / s)
    else:
        # Simple overlapping block bootstrap on trade index
        max_start = N - block_len
        for _ in range(int(n_boot)):
            n_blocks = int(np.ceil(N / block_len))
            pieces = []
            for _ in range(n_blocks):
                start = np.random.randint(0, max_start + 1)
                pieces.append(vals[start : start + block_len])
            sample = np.concatenate(pieces)[:N]
            m = sample.mean()
            s = sample.std(ddof=1)
            if s > 0:
                sharpes.append(m / s)

    return np.array(sharpes, dtype=float)


def _of_mc_bootstrap_paths(
    vals,
    n_mc=500,
    early_frac=0.25,
    max_paths=OF_MAX_FAN_PATHS,
):
    """
    Monte Carlo path bootstrap WITH replacement on trades.

    For each MC run:
      - sample N trades with replacement
      - compute equity path
      - compute max DD, final cumulative, loss-streak stats
      - track whether a loss streak >= actual_max_streak appears
        in the first 10% and 25% of the path
      - track the earliest trade index where such a streak appears
        anywhere in the path (if ever)

    Returns:
        dd_arr          : max DD for each path
        final_arr       : final cumulative for each path
        max_st_arr      : max loss streak for each path
        avg_st_arr      : avg loss streak (len>=2) for each path
        prob_early_10   : P[loss streak ≥ actual in first 10% trades]
        prob_early_25   : P[loss streak ≥ actual in first 25% trades]
        first_frac_arr  : array of fractions in [0,1], location of first
                          such streak as a fraction of path length, for
                          paths where it occurs at least once
        fan_paths       : array of shape (K, N) with up to max_paths equity paths
    """
    vals = np.asarray(vals, dtype=float)
    N = vals.size
    if N == 0 or n_mc is None or n_mc <= 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            0.0,
            0.0,
            np.array([]),
            np.empty((0, 0)),
        )

    n_mc = int(n_mc)
    early_len_25 = max(1, int(round(early_frac * N)))
    early_len_10 = max(1, int(round(0.10 * N)))

    # Actual streak on historical series
    actual_max_streak, _ = _of_compute_losing_streak_stats(vals)

    dd_list = []
    final_list = []
    max_streak_list = []
    avg_long_list = []
    early_hits_10 = 0
    early_hits_25 = 0
    first_frac_list = []

    fan_paths = []

    for i in range(n_mc):
        sample = np.random.choice(vals, size=N, replace=True)
        equity = np.cumsum(sample)
        dd = _of_compute_max_drawdown_from_series(sample)
        dd_list.append(dd)
        final_list.append(float(equity[-1]))

        ms, avg_long = _of_compute_losing_streak_stats(sample)
        max_streak_list.append(ms)
        avg_long_list.append(avg_long)

        if actual_max_streak > 0:
            # 10% and 25% early windows
            def has_streak_in_first(M):
                losses = sample[:M] < 0
                cur = 0
                for is_loss in losses:
                    if is_loss:
                        cur += 1
                        if cur >= actual_max_streak:
                            return True
                    else:
                        cur = 0
                return False

            if has_streak_in_first(early_len_10):
                early_hits_10 += 1
            if has_streak_in_first(early_len_25):
                early_hits_25 += 1

            # Earliest occurrence anywhere in the path
            losses_full = sample < 0
            cur = 0
            earliest_idx = None
            for idx, is_loss in enumerate(losses_full):
                if is_loss:
                    cur += 1
                    if cur >= actual_max_streak:
                        # index where the streak starts
                        start_idx = idx - actual_max_streak + 1
                        earliest_idx = start_idx
                        break
                else:
                    cur = 0
            if earliest_idx is not None:
                first_frac_list.append(earliest_idx / N)

        if i < max_paths:
            fan_paths.append(equity.copy())

    dd_arr = np.array(dd_list, dtype=float)
    final_arr = np.array(final_list, dtype=float)
    max_st_arr = np.array(max_streak_list, dtype=int)
    avg_st_arr = np.array(avg_long_list, dtype=float)

    prob_early_10 = (
        (early_hits_10 / n_mc)
        if n_mc > 0 and actual_max_streak > 0
        else 0.0
    )
    prob_early_25 = (
        (early_hits_25 / n_mc)
        if n_mc > 0 and actual_max_streak > 0
        else 0.0
    )

    first_frac_arr = (
        np.array(first_frac_list, dtype=float)
        if first_frac_list
        else np.array([], dtype=float)
    )

    fan_paths = np.vstack(fan_paths) if fan_paths else np.empty((0, N))

    return (
        dd_arr,
        final_arr,
        max_st_arr,
        avg_st_arr,
        prob_early_10,
        prob_early_25,
        first_frac_arr,
        fan_paths,
    )


def _fmt_float(x, digits=3):
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "-"
    except Exception:
        pass
    return f"{float(x):.{digits}f}"


def _fmt_dollar(x):
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "-"
    except Exception:
        pass
    return f"${float(x):,.1f}"


def _fmt_pct(x, digits=2):
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "-"
    except Exception:
        pass
    return f"{float(x) * 100:.{digits}f}%"



@callback(
    Output("p1-overfit-bootstrap-hist", "figure"),
    Output("p1-overfit-metrics-bootstrap", "children"),
    Input("p1-overfit-run-bootstrap", "n_clicks"),
    State("p1-strategy-checklist", "value"),
    State("p1-overfit-metric-mode", "value"),
    State("p1-overfit-bootstrap-n", "value"),
    State("p1-overfit-bootstrap-block", "value"),
)
def update_p1_overfit_bootstrap(
    n_clicks,
    selected_strategy_ids,
    metric_mode,
    n_boot,
    block_len,
):
    def empty_fig():
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            title={
                "text": "Bootstrap Sharpe distribution",
                "x": 0.01,
                "xanchor": "left",
            },
        )
        return fig

    if not n_clicks:
        return empty_fig(), html.Div(
            "Press 'Run bootstrap' to compute Sharpe distribution.",
            style={"color": "#AAAAAA"},
        )

    prep = _of_prepare_series(selected_strategy_ids, metric_mode)
    if prep is None or prep["n_trades"] < 10:
        return empty_fig(), html.Div(
            "Not enough trades for bootstrap (need at least 10).",
            style={"color": "#AAAAAA"},
        )

    # Original per-trade Sharpe
    # vals = prep["series"]
    # metric_label = prep["metric_label"]
    # # Baseline Sharpe
    # mean_val = float(vals.mean())
    # std_val = float(vals.std(ddof=1)) if prep["n_trades"] > 1 else 0.0
    # sharpe = (mean_val / std_val) if std_val > 0 else np.nan
    
    vals = prep["series"]
    metric_label = prep["metric_label"]
    
    # Baseline Sharpe (annualised, treating each trade as one period)
    mean_val = float(vals.mean())
    std_val = float(vals.std(ddof=1)) if prep["n_trades"] > 1 else 0.0
    if std_val > 0:
        sharpe = (mean_val / std_val) * (252 ** 0.5)
    else:
        sharpe = np.nan

    # Inputs
    try:
        n_boot = int(n_boot) if n_boot is not None else 500
    except Exception:
        n_boot = 500
    if n_boot < 100:
        n_boot = 100

    try:
        block_len = int(block_len) if block_len is not None else 1
    except Exception:
        block_len = 1

    boot_sharpes = _of_bootstrap_sharpes(vals, n_boot=n_boot, block_len=block_len)

    if boot_sharpes.size == 0:
        fig = empty_fig()
        prob_lt0 = np.nan
        q5 = q50 = q95 = np.nan
    # Original per-trade Sharpe
    # else:
    #     prob_lt0 = float((boot_sharpes < 0).mean())
    #     q5, q50, q95 = np.percentile(boot_sharpes, [5, 50, 95])

    #     hist_counts, hist_edges = np.histogram(boot_sharpes, bins=30)
    
    else:
        # Annualise the bootstrap Sharpe values using the same √252 factor
        boot_sharpes = boot_sharpes * (252 ** 0.5)
    
        prob_lt0 = float((boot_sharpes < 0).mean())
        q5, q50, q95 = np.percentile(boot_sharpes, [5, 50, 95])
        hist_counts, hist_edges = np.histogram(boot_sharpes, bins=30)


        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=boot_sharpes,
                nbinsx=30,
                name="Bootstrap Sharpe",
                opacity=0.8,
            )
        )
        if not np.isnan(sharpe):
            fig.add_trace(
                go.Scatter(
                    x=[sharpe, sharpe],
                    y=[0, max(1, hist_counts.max())],
                    mode="lines",
                    name="Actual Sharpe",
                )
            )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            title={
                "text": "Bootstrap Sharpe distribution",
                "x": 0.01,
                "xanchor": "left",
            },
            xaxis={"title": "Sharpe"},
            yaxis={"title": "Frequency"},
            barmode="overlay",
        )

    # Choose formatter for the mean depending on metric
    if metric_label.startswith("P&L"):
        mean_str = _fmt_dollar(mean_val)
    else:
        mean_str = _fmt_float(mean_val, 3)
        
    table_style = {
        "marginTop": "0.25rem",
        "tableLayout": "fixed",
        "width": "100%",
        }
        
    table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Metric", style={"width": "55%"}),
                        html.Th("Value", style={"width": "45%"}),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td("Trades (N)"),
                            html.Td(str(prep["n_trades"])),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(f"Mean {metric_label}"),
                            html.Td(mean_str),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Sharpe (actual)"),
                            html.Td(_fmt_float(sharpe, 3)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Bootstrap runs"),
                            html.Td(str(n_boot)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Block length"),
                            html.Td(str(block_len)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("P(Sharpe < 0)"),
                            html.Td(_fmt_pct(prob_lt0, 2)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Sharpe 5 / 50 / 95 pct"),
                            html.Td(
                                f"{_fmt_float(q5,3)} / "
                                f"{_fmt_float(q50,3)} / "
                                f"{_fmt_float(q95,3)}"
                            ),
                        ]
                    ),
                ]
            ),
        ],
        bordered=True,
        hover=True,
        striped=True,
        size="sm",
        style=table_style,
    )


    return fig, table

@callback(
    Output("p1-overfit-topk-bar", "figure"),
    Output("p1-overfit-metrics-topk", "children"),
    Input("p1-overfit-topk", "value"),
    Input("p1-strategy-checklist", "value"),
    Input("p1-overfit-metric-mode", "value"),
)
def update_p1_overfit_topk(
    top_k,
    selected_strategy_ids,
    metric_mode,
):
    def empty_fig():
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            title={
                "text": "Effect of removing top-K trades",
                "x": 0.01,
                "xanchor": "left",
            },
        )
        return fig

    prep = _of_prepare_series(selected_strategy_ids, metric_mode)
    if prep is None or prep["n_trades"] < 2:
        return empty_fig(), html.Div(
            "Not enough trades for Top-K analysis.",
            style={"color": "#AAAAAA"},
        )

    vals = prep["series"]
    metric_label = prep["metric_label"]
    unit_label = prep["unit_label"]
    n_trades = prep["n_trades"]

    # Baseline stats
    mean_val = float(vals.mean())
    std_val = float(vals.std(ddof=1)) if n_trades > 1 else 0.0
    sharpe = (mean_val / std_val) if std_val > 0 else np.nan
    total_val = float(vals.sum())
    max_dd_actual = _of_compute_max_drawdown_from_series(vals)

    # Top-K removal
    try:
        top_k = int(top_k) if top_k is not None else 3
    except Exception:
        top_k = 3
    if top_k < 0:
        top_k = 0
    if top_k > n_trades - 1:
        top_k = max(0, n_trades - 1)

    vals_sorted = np.sort(vals)[::-1]  # descending
    if top_k > 0:
        vals_after = vals_sorted[top_k:]
        topk_sum = float(vals_sorted[:top_k].sum())
    else:
        vals_after = vals_sorted.copy()
        topk_sum = 0.0

    mean_after = float(vals_after.mean())
    std_after = float(vals_after.std(ddof=1)) if vals_after.size > 1 else 0.0
    sharpe_after = (mean_after / std_after) if std_after > 0 else np.nan
    max_dd_after = _of_compute_max_drawdown_from_series(vals_after)

    if abs(total_val) > 1e-8 and top_k > 0:
        pct_contrib = float(100.0 * topk_sum / total_val)
    else:
        pct_contrib = np.nan

    # For plotting, map NaN Sharpe to 0 so bars don't disappear
    sharpe_plot = 0.0 if np.isnan(sharpe) else sharpe
    sharpe_after_plot = 0.0 if np.isnan(sharpe_after) else sharpe_after

    fig = go.Figure()

    # We use two offsetgroups: "baseline" and "minus"
    # so that bars are grouped side-by-side for both Mean and Sharpe.

    # Mean (left axis)
    fig.add_trace(
        go.Bar(
            x=["Mean"],
            y=[mean_val],
            name="Baseline mean",
            offsetgroup="baseline",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Bar(
            x=["Mean"],
            y=[mean_after],
            name=f"Minus top {top_k} mean",
            offsetgroup="minus",
            yaxis="y1",
        )
    )

    # Sharpe (right axis)
    fig.add_trace(
        go.Bar(
            x=["Sharpe"],
            y=[sharpe_plot],
            name="Baseline Sharpe",
            offsetgroup="baseline",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Bar(
            x=["Sharpe"],
            y=[sharpe_after_plot],
            name=f"Minus top {top_k} Sharpe",
            offsetgroup="minus",
            yaxis="y2",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        title={
            "text": "Effect of removing top-K trades",
            "x": 0.01,
            "xanchor": "left",
        },
        xaxis={"title": "Metric"},
        yaxis=dict(
            title=f"Mean {metric_label}",
            side="left",
        ),
        yaxis2=dict(
            title="Sharpe",
            overlaying="y",
            side="right",
        ),
        barmode="group",
        bargap=0.25,       # space between Mean and Sharpe groups
        bargroupgap=0.1,   # space inside each group (baseline vs minus)
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.15,
            yanchor="top",
        ),
    )

    if metric_label.startswith("P&L"):
        mean_base_str = _fmt_dollar(mean_val)
        mean_after_str = _fmt_dollar(mean_after)
        dd_base_str = _fmt_dollar(max_dd_actual)
        dd_after_str = _fmt_dollar(max_dd_after)
    else:
        mean_base_str = _fmt_float(mean_val, 3)
        mean_after_str = _fmt_float(mean_after, 3)
        dd_base_str = _fmt_float(max_dd_actual, 3)
        dd_after_str = _fmt_float(max_dd_after, 3)
        
        
    table_style = {
        "marginTop": "0.25rem",
        "tableLayout": "fixed",
        "width": "100%",
        }

    table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Metric", style={"width": "55%"}),
                        html.Th("Value", style={"width": "45%"}),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td("Trades (N)"),
                            html.Td(str(n_trades)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(f"Mean {metric_label} (baseline)"),
                            html.Td(mean_base_str),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Sharpe (baseline)"),
                            html.Td(_fmt_float(sharpe, 3)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Max DD (baseline)"),
                            html.Td(dd_base_str),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(f"Mean {metric_label} (minus top {top_k})"),
                            html.Td(mean_after_str),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(f"Sharpe (minus top {top_k})"),
                            html.Td(_fmt_float(sharpe_after, 3)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(f"Max DD (minus top {top_k})"),
                            html.Td(dd_after_str),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(f"% total result from top {top_k}"),
                            html.Td(
                                "-"
                                if np.isnan(pct_contrib)
                                else _fmt_pct(pct_contrib / 100.0, 1)
                            ),
                        ]
                    ),
                ]
            ),
        ],
        bordered=True,
        hover=True,
        striped=True,
        size="sm",
        style=table_style,
    )


    return fig, table

@callback(
    Output("p1-overfit-mc-dd-hist", "figure"),
    Output("p1-overfit-mc-final-hist", "figure"),
    Output("p1-overfit-mc-fan", "figure"),
    Output("p1-overfit-metrics-mc", "children"),
    Input("p1-overfit-run-mc", "n_clicks"),
    State("p1-strategy-checklist", "value"),
    State("p1-overfit-metric-mode", "value"),
    State("p1-overfit-mc-n", "value"),
)
def update_p1_overfit_mc(
    n_clicks,
    selected_strategy_ids,
    metric_mode,
    n_mc,
):
    def empty_dd_fig():
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            title={
                "text": "MC distribution of max drawdown",
                "x": 0.01,
                "xanchor": "left",
            },
        )
        return fig

    def empty_final_fig():
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            title={
                "text": "MC distribution of final cumulative result",
                "x": 0.01,
                "xanchor": "left",
            },
        )
        return fig

    def empty_fan_fig():
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222222",
            plot_bgcolor="#222222",
            font={"color": "#EEEEEE"},
            title={
                "text": "MC bootstrapped equity paths",
                "x": 0.01,
                "xanchor": "left",
            },
        )
        return fig

    if not n_clicks:
        return (
            empty_dd_fig(),
            empty_final_fig(),
            empty_fan_fig(),
            html.Div(
                "Press 'Run MC' to generate Monte Carlo paths.",
                style={"color": "#AAAAAA"},
            ),
        )

    prep = _of_prepare_series(selected_strategy_ids, metric_mode)
    if prep is None or prep["n_trades"] < 10:
        return (
            empty_dd_fig(),
            empty_final_fig(),
            empty_fan_fig(),
            html.Div(
                "Not enough trades for MC path bootstrap (need at least 10).",
                style={"color": "#AAAAAA"},
            ),
        )

    vals = prep["series"]
    metric_label = prep["metric_label"]
    unit_label = prep["unit_label"]
    n_trades = prep["n_trades"]
    equity_actual = prep["equity_actual"]

    # Baseline stats
    mean_val = float(vals.mean())
    std_val = float(vals.std(ddof=1)) if n_trades > 1 else 0.0
    sharpe = (mean_val / std_val) if std_val > 0 else np.nan
    total_val = float(vals.sum())
    max_dd_actual = _of_compute_max_drawdown_from_series(vals)
    max_streak_actual, avg_streak_actual = _of_compute_losing_streak_stats(vals)

    # MC settings
    try:
        n_mc = int(n_mc) if n_mc is not None else 500
    except Exception:
        n_mc = 500
    if n_mc < 100:
        n_mc = 100

    (
        dd_arr,
        final_arr,
        max_st_arr,
        avg_st_arr,
        prob_early_10,
        prob_early_25,
        first_frac_arr,
        fan_paths,
    ) = _of_mc_bootstrap_paths(vals, n_mc=n_mc, early_frac=0.25)


    if dd_arr.size == 0:
        return (
            empty_dd_fig(),
            empty_final_fig(),
            empty_fan_fig(),
            html.Div(
                "MC path bootstrap failed (no samples).",
                style={"color": "#AAAAAA"},
            ),
        )

    # Quantiles
    dd_q5, dd_q50, dd_q95 = np.percentile(dd_arr, [5, 50, 95])
    final_q5, final_q50, final_q95 = np.percentile(final_arr, [5, 50, 95])
    
    # First-disaster location quantiles (fraction of path)
    if first_frac_arr.size > 0:
        frac_q5, frac_q50, frac_q95 = np.percentile(first_frac_arr, [5, 50, 95])
    else:
        frac_q5 = frac_q50 = frac_q95 = np.nan

    max_st_q50 = float(np.median(max_st_arr))
    avg_st_q50 = float(np.median(avg_st_arr))

    # DD histogram
    dd_hist_counts, _ = np.histogram(dd_arr, bins=30)
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Histogram(
            x=dd_arr,
            nbinsx=30,
            name="MC max DD",
            opacity=0.8,
        )
    )
    fig_dd.add_trace(
        go.Scatter(
            x=[max_dd_actual, max_dd_actual],
            y=[0, max(1, dd_hist_counts.max())],
            mode="lines",
            name="Actual max DD",
        )
    )
    fig_dd.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        title={
            "text": "MC distribution of max drawdown",
            "x": 0.01,
            "xanchor": "left",
        },
        xaxis={"title": f"Max DD ({unit_label})"},
        yaxis={"title": "Frequency"},
        barmode="overlay",
    )

    # Final cumulative histogram
    final_hist_counts, _ = np.histogram(final_arr, bins=30)
    fig_final = go.Figure()
    fig_final.add_trace(
        go.Histogram(
            x=final_arr,
            nbinsx=30,
            name="MC final cumulative",
            opacity=0.8,
        )
    )
    fig_final.add_trace(
        go.Scatter(
            x=[total_val, total_val],
            y=[0, max(1, final_hist_counts.max())],
            mode="lines",
            name="Actual cumulative",
        )
    )
    fig_final.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        title={
            "text": "MC distribution of final cumulative result",
            "x": 0.01,
            "xanchor": "left",
        },
        xaxis={"title": f"Final cumulative ({unit_label})"},
        yaxis={"title": "Frequency"},
        barmode="overlay",
    )

    # Fan chart
    fig_fan = go.Figure()
    x_vals = list(range(1, n_trades + 1))

    if fan_paths.size > 0:
        for path in fan_paths:
            fig_fan.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=path,
                    mode="lines",
                    line={"width": 1},
                    opacity=0.2,
                    showlegend=False,
                )
            )

        # Median path
        median_path = np.median(fan_paths, axis=0)
        fig_fan.add_trace(
            go.Scatter(
                x=x_vals,
                y=median_path,
                mode="lines",
                line={"width": 2, "color": "red"},
                name="MC median",
            )
        )

    # Actual equity
    fig_fan.add_trace(
        go.Scatter(
            x=x_vals,
            y=equity_actual,
            mode="lines",
            line={"width": 2, "color": "white"},
            name="Actual",
        )
    )
    fig_fan.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        title={
            "text": "MC bootstrapped equity paths",
            "x": 0.01,
            "xanchor": "left",
        },
        xaxis={"title": "Trade index"},
        yaxis={"title": f"Cumulative {metric_label}"},
    )

    # Choose formatters depending on metric
    if metric_label.startswith("P&L"):
        mean_str = _fmt_dollar(mean_val)
        total_str = _fmt_dollar(total_val)
        dd_actual_str = _fmt_dollar(max_dd_actual)
        dd_q_str = (
            f"{_fmt_dollar(dd_q5)} / {_fmt_dollar(dd_q50)} / {_fmt_dollar(dd_q95)}"
        )
        final_q_str = (
            f"{_fmt_dollar(final_q5)} / "
            f"{_fmt_dollar(final_q50)} / "
            f"{_fmt_dollar(final_q95)}"
        )
    else:
        mean_str = _fmt_float(mean_val, 3)
        total_str = _fmt_float(total_val, 3)
        dd_actual_str = _fmt_float(max_dd_actual, 3)
        dd_q_str = (
            f"{_fmt_float(dd_q5,3)} / "
            f"{_fmt_float(dd_q50,3)} / "
            f"{_fmt_float(dd_q95,3)}"
        )
        final_q_str = (
            f"{_fmt_float(final_q5,3)} / "
            f"{_fmt_float(final_q50,3)} / "
            f"{_fmt_float(final_q95,3)}"
        )

    
    table_style = {
        "marginTop": "0.25rem",
        "tableLayout": "fixed",
        "width": "100%",
        }

    table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Metric", style={"width": "55%"}),
                        html.Th("Value", style={"width": "45%"}),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td("Trades (N)"),
                            html.Td(str(n_trades)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(f"Mean {metric_label}"),
                            html.Td(mean_str),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Sharpe (actual)"),
                            html.Td(_fmt_float(sharpe, 3)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(f"Total {metric_label} (actual)"),
                            html.Td(total_str),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(f"Max DD actual ({unit_label})"),
                            html.Td(dd_actual_str),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("MC permutations"),
                            html.Td(str(n_mc)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("MC max DD 5 / 50 / 95 pct"),
                            html.Td(dd_q_str),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("MC final 5 / 50 / 95 pct"),
                            html.Td(final_q_str),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Max loss streak (actual)"),
                            html.Td(str(max_streak_actual)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Avg loss streak (actual, len≥2)"),
                            html.Td(_fmt_float(avg_streak_actual, 3)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Median max loss streak (MC)"),
                            html.Td(_fmt_float(max_st_q50, 3)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Median avg loss streak (MC, len≥2)"),
                            html.Td(_fmt_float(avg_st_q50, 3)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(
                                "P[loss streak ≥ actual within first 10% trades]"
                            ),
                            html.Td(_fmt_pct(prob_early_10, 3)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(
                                "P[loss streak ≥ actual within first 25% trades]"
                            ),
                            html.Td(_fmt_pct(prob_early_25, 3)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(
                                "First disastrous streak location "
                                "(5 / 50 / 95 pct of trades)"
                            ),
                            html.Td(
                                f"{_fmt_pct(frac_q5, 1)} / "
                                f"{_fmt_pct(frac_q50, 1)} / "
                                f"{_fmt_pct(frac_q95, 1)}"
                            ),
                        ]
                    ),

                ]
            ),
        ],
        bordered=True,
        hover=True,
        striped=True,
        size="sm",
        style=table_style,
    )


    return fig_dd, fig_final, fig_fan, table
