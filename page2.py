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
from dash import html, dcc, callback, Input, Output, State, ctx, no_update

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

from core.sh_layout import (
    build_data_input_section,
    build_strategy_sidebar,
    ROOT_DATA_DIR,
    strategy_color_for_uid,
    _list_immediate_subfolders,
    p1_strategy_store,   # reuse in-memory strategy data (with df)
)
from core.registry import list_portfolios
from pages.page1 import VIX_SLIDER_MIN, VIX_SLIDER_MAX, VIX_MARKS



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
                                children=html.Div(
                                    [
                                        # ------- Top controls + metrics ---------------------------------
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Span(
                                                                "Initial equity:",
                                                                style={
                                                                    "marginRight": "0.5rem",
                                                                    "fontSize": "0.85rem",
                                                                },
                                                            ),
                                                            dcc.Input(
                                                                id="p2-initial-equity-input",
                                                                type="number",
                                                                value=100000,
                                                                min=0,
                                                                step=10000,
                                                                style={
                                                                    "width": "140px",
                                                                    "fontSize": "0.85rem",
                                                                    "backgroundColor": "#2a2a2a",
                                                                    "color": "#EEEEEE",
                                                                    "border": "1px solid #555555",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "display": "flex",
                                                            "alignItems": "center",
                                                            "gap": "0.5rem",
                                                        },
                                                    ),
                                                    md=12,
                                                ),
                                            ],
                                            className="mb-2",
                                        ),

                                        # Metrics table / summary for portfolio
                                        html.Div(
                                            id="p2-portfolio-metrics",
                                            style={
                                                "fontSize": "0.85rem",
                                                "marginBottom": "0.75rem",
                                            },
                                        ),
                                        html.Hr(style={"marginTop": "0.25rem", "marginBottom": "0.75rem"}),
                                        
                                        html.Hr(style={"marginTop": "0.25rem", "marginBottom": "0.75rem"}),

                                        # Toggle for strategy overlays on equity chart
                                        html.Div(
                                            dbc.Checkbox(
                                                id="p2-show-strategy-equity",
                                                value=False,
                                                label="Overlay individual strategy equity curves",
                                                label_style={"fontSize": "0.8rem"},
                                            ),
                                            style={"marginBottom": "0.35rem"},
                                        ),
                                                                   
                                        # ------- Middle: Equity & Drawdown charts (stacked full-width) --
                                        
                                        
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-equity-graph",
                                                        figure={
                                                            "data": [],
                                                            "layout": {
                                                                "template": "plotly_dark",
                                                                "paper_bgcolor": "#222222",
                                                                "plot_bgcolor": "#222222",
                                                                "font": {"color": "#EEEEEE"},
                                                            },
                                                        },
                                                        style={"height": "320px"},
                                                    ),
                                                    md=12,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-dd-graph",
                                                        figure={
                                                            "data": [],
                                                            "layout": {
                                                                "template": "plotly_dark",
                                                                "paper_bgcolor": "#222222",
                                                                "plot_bgcolor": "#222222",
                                                                "font": {"color": "#EEEEEE"},
                                                            },
                                                        },
                                                        style={"height": "260px"},
                                                    ),
                                                    md=12,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),

                            
                                        # ------- Bottom: Histogram + DOW exposure -----------------------
                                        dbc.Row(
                                            [
                                                # Left: P&L histogram + distribution metrics
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                id="p2-dist-metrics",
                                                                style={
                                                                    "fontSize": "0.8rem",
                                                                    "marginBottom": "0.4rem",
                                                                },
                                                            ),
                                                            dcc.Graph(
                                                                id="p2-pnl-histogram",
                                                                figure={
                                                                    "data": [],
                                                                    "layout": {
                                                                        "template": "plotly_dark",
                                                                        "paper_bgcolor": "#222222",
                                                                        "plot_bgcolor": "#222222",
                                                                        "font": {"color": "#EEEEEE"},
                                                                    },
                                                                },
                                                                style={"height": "260px"},
                                                            ),
                                                        ]
                                                    ),
                                                    md=6,
                                                ),
                                                # Right: Day-of-week exposure chart
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-dow-bar-graph",
                                                        figure={
                                                            "data": [],
                                                            "layout": {
                                                                "template": "plotly_dark",
                                                                "paper_bgcolor": "#222222",
                                                                "plot_bgcolor": "#222222",
                                                                "font": {"color": "#EEEEEE"},
                                                            },
                                                        },
                                                        style={"height": "260px"},
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="mb-2",
                                        ),
                                        #here
                                        # ---------------- Row: P&L Contribution Waterfall -------------------
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "P&L Contribution – Waterfall",
                                                                        style={
                                                                            "fontSize": "0.8rem",
                                                                            "fontWeight": "bold",
                                                                            "color": "#DDDDDD",
                                                                        },
                                                                    ),
                                                                    html.Span(
                                                                        " ⓘ",
                                                                        title=(
                                                                            "Shows total weighted P&L contribution "
                                                                            "of each strategy to the portfolio. "
                                                                            "Toggle between absolute $ and % of total."
                                                                        ),
                                                                        style={
                                                                            "marginLeft": "0.35rem",
                                                                            "cursor": "help",
                                                                            "fontSize": "0.8rem",
                                                                            "color": "#AAAAAA",
                                                                        },
                                                                    ),
                                                                    dcc.RadioItems(
                                                                        id="p2-contr-pnl-mode",
                                                                        options=[
                                                                            {"label": " ABS $", "value": "abs"},
                                                                            {"label": " % of total", "value": "pct"},
                                                                        ],
                                                                        value="abs",
                                                                        inline=True,
                                                                        labelStyle={"marginRight": "0.75rem"},
                                                                        style={
                                                                            "marginLeft": "1.0rem",
                                                                            "fontSize": "0.85rem",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "alignItems": "center",
                                                                    "gap": "0.5rem",
                                                                    "marginBottom": "0.35rem",
                                                                },
                                                            ),
                                                            dcc.Graph(
                                                                id="p2-contr-pnl-waterfall",
                                                                figure=_empty_corr_figure(
                                                                    "Select strategies to see P&L contribution."
                                                                ),
                                                                style={"height": "340px"},
                                                            ),
                                                        ]
                                                    ),
                                                    md=12,
                                                ),
                                            ],
                                            className="mb-2",
                                        ),
                                        
                                        # ---------------- Row: DD Contribution Bars -------------------
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "DD Contribution – Weighted (DD-worsening days only)",
                                                                        style={
                                                                            "fontSize": "0.8rem",
                                                                            "fontWeight": "bold",
                                                                            "color": "#DDDDDD",
                                                                        },
                                                                    ),
                                                                    html.Span(
                                                                        " ⓘ",
                                                                        title=(
                                                                            "Aggregates each strategy’s weighted P&L "
                                                                            "only on days where the PORTFOLIO drawdown increases. "
                                                                            "Highlights true DD drivers vs stabilisers."
                                                                        ),
                                                                        style={
                                                                            "marginLeft": "0.35rem",
                                                                            "cursor": "help",
                                                                            "fontSize": "0.8rem",
                                                                            "color": "#AAAAAA",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "alignItems": "center",
                                                                    "gap": "0.5rem",
                                                                    "marginBottom": "0.35rem",
                                                                },
                                                            ),
                                                            dcc.Graph(
                                                                id="p2-contr-dd-bars",
                                                                figure=_empty_corr_figure(
                                                                    "Select strategies to see DD contributions."
                                                                ),
                                                                style={"height": "340px"},
                                                            ),
                                                        ]
                                                    ),
                                                    md=12,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),


                                    ],
                                    style={
                                        "padding": "0.75rem",
                                        "fontSize": "0.85rem",
                                    },
                                ),
                            ),

                            # ------------------------------------------------------------------
                            # Phase 2 – Correlation tab (CORR1–CORR4)
                            # ------------------------------------------------------------------
                            dcc.Tab(
                                label="Correlation",
                                value="p2-corr",
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
                                children=html.Div(
                                    [
                                        # Short header note
                                        html.Div(
                                            "Correlation metrics are computed on daily P&L of the "
                                            "selected strategies in the Active list.",
                                            style={
                                                "fontSize": "0.8rem",
                                                "marginBottom": "0.5rem",
                                                "color": "#CCCCCC",
                                            },
                                        ),
                    
                                        # ---------------- Row 1: CORR1 & CORR2 -------------------
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            # CORR1 label + tooltip
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "CORR1 – Pearson correlation (daily P&L)",
                                                                        style={
                                                                            "fontSize": "0.8rem",
                                                                            "fontWeight": "bold",
                                                                            "color": "#DDDDDD",
                                                                        },
                                                                    ),
                                                                    html.Span(
                                                                        " ⓘ",
                                                                        title=(
                                                                            "Standard Pearson correlation between daily P&L "
                                                                            "series of each pair of strategies. "
                                                                            "1 = move together, 0 = uncorrelated, -1 = move opposite."
                                                                        ),
                                                                        style={
                                                                            "marginLeft": "0.35rem",
                                                                            "cursor": "help",
                                                                            "fontSize": "0.8rem",
                                                                            "color": "#AAAAAA",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "alignItems": "center",
                                                                    "marginBottom": "0.25rem",
                                                                },
                                                            ),
                                                            dcc.Graph(
                                                                id="p2-corr-pearson-heatmap",
                                                                figure={
                                                                    "data": [],
                                                                    "layout": {
                                                                        "template": "plotly_dark",
                                                                        "paper_bgcolor": "#222222",
                                                                        "plot_bgcolor": "#222222",
                                                                        "font": {"color": "#EEEEEE"},
                                                                    },
                                                                },
                                                                style={"height": "320px"},
                                                            ),
                                                        ]
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            # CORR2 label + tooltip
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "CORR2 – Downside correlation",
                                                                        style={
                                                                            "fontSize": "0.8rem",
                                                                            "fontWeight": "bold",
                                                                            "color": "#DDDDDD",
                                                                        },
                                                                    ),
                                                                    html.Span(
                                                                        " ⓘ",
                                                                        title=(
                                                                            "Pearson correlation computed only on days when "
                                                                            "both strategies have negative daily P&L. "
                                                                            "Highlights co-movement on losing days."
                                                                        ),
                                                                        style={
                                                                            "marginLeft": "0.35rem",
                                                                            "cursor": "help",
                                                                            "fontSize": "0.8rem",
                                                                            "color": "#AAAAAA",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "alignItems": "center",
                                                                    "marginBottom": "0.25rem",
                                                                },
                                                            ),
                                                            dcc.Graph(
                                                                id="p2-corr-downside-heatmap",
                                                                figure={
                                                                    "data": [],
                                                                    "layout": {
                                                                        "template": "plotly_dark",
                                                                        "paper_bgcolor": "#222222",
                                                                        "plot_bgcolor": "#222222",
                                                                        "font": {"color": "#EEEEEE"},
                                                                    },
                                                                },
                                                                style={"height": "320px"},
                                                            ),
                                                        ]
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                    
                                        # ---------------- Row 2: CORR3 & CORR4 -------------------
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            # CORR3 label + tooltip
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "CORR3 – Tail co-crash frequency (5% tails)",
                                                                        style={
                                                                            "fontSize": "0.8rem",
                                                                            "fontWeight": "bold",
                                                                            "color": "#DDDDDD",
                                                                        },
                                                                    ),
                                                                    html.Span(
                                                                        " ⓘ",
                                                                        title=(
                                                                            "Empirical probability that both strategies are in their "
                                                                            "worst 5% daily P&L days at the same time. "
                                                                            "0 = never, 1 = always."
                                                                        ),
                                                                        style={
                                                                            "marginLeft": "0.35rem",
                                                                            "cursor": "help",
                                                                            "fontSize": "0.8rem",
                                                                            "color": "#AAAAAA",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "alignItems": "center",
                                                                    "marginBottom": "0.25rem",
                                                                },
                                                            ),
                                                            dcc.Graph(
                                                                id="p2-corr-taildep-heatmap",
                                                                figure={
                                                                    "data": [],
                                                                    "layout": {
                                                                        "template": "plotly_dark",
                                                                        "paper_bgcolor": "#222222",
                                                                        "plot_bgcolor": "#222222",
                                                                        "font": {"color": "#EEEEEE"},
                                                                    },
                                                                },
                                                                style={"height": "260px"},
                                                            ),
                                                        ]
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            # CORR4 label + tooltip
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "CORR4 – Rank correlation summary (Kendall / Spearman)",
                                                                        style={
                                                                            "fontSize": "0.8rem",
                                                                            "fontWeight": "bold",
                                                                            "color": "#DDDDDD",
                                                                        },
                                                                    ),
                                                                    html.Span(
                                                                        " ⓘ",
                                                                        title=(
                                                                            "Average Kendall tau and Spearman rank correlation of each "
                                                                            "strategy versus all others, based on daily P&L ranks."
                                                                        ),
                                                                        style={
                                                                            "marginLeft": "0.35rem",
                                                                            "cursor": "help",
                                                                            "fontSize": "0.8rem",
                                                                            "color": "#AAAAAA",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "alignItems": "center",
                                                                    "marginBottom": "0.25rem",
                                                                },
                                                            ),
                                                            html.Div(
                                                                id="p2-corr-alt-table",
                                                                style={
                                                                    "fontSize": "0.8rem",
                                                                    "maxHeight": "260px",
                                                                    "overflowY": "auto",
                                                                    "backgroundColor": "#222222",
                                                                    "border": "1px solid #444444",
                                                                    "padding": "0.4rem",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="mb-2",
                                        ),
                                        # ---------------- Row 3: CORR5 – Strategy vs Portfolio by VIX Regime -------------------
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "CORR5 – Strategy vs Portfolio correlation by VIX regime",
                                                                        style={
                                                                            "fontSize": "0.8rem",
                                                                            "fontWeight": "bold",
                                                                            "color": "#DDDDDD",
                                                                        },
                                                                    ),
                                                                    html.Span(
                                                                        " ⓘ",
                                                                        title=(
                                                                            "Correlation of each strategy's daily P&L with the portfolio daily P&L "
                                                                            "computed separately within each VIX quartile bucket (entry-based Opening VIX)."
                                                                        ),
                                                                        style={
                                                                            "marginLeft": "0.35rem",
                                                                            "cursor": "help",
                                                                            "fontSize": "0.8rem",
                                                                            "color": "#AAAAAA",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "alignItems": "center",
                                                                    "marginBottom": "0.25rem",
                                                                },
                                                            ),
                                                            # --- CORR5 VIX mode controls (AUTO / MANUAL) ---
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "VIX regime bucketing:",
                                                                        style={
                                                                            "fontSize": "0.75rem",
                                                                            "color": "#CCCCCC",
                                                                            "marginRight": "0.5rem",
                                                                        },
                                                                    ),
                                                                    dbc.ButtonGroup(
                                                                        [
                                                                            dbc.Button(
                                                                                "AUTO",
                                                                                id="p2-corr5-vix-mode-auto",
                                                                                n_clicks=1,
                                                                                size="sm",
                                                                                color="secondary",
                                                                                outline=False,
                                                                            ),
                                                                            dbc.Button(
                                                                                "MANUAL",
                                                                                id="p2-corr5-vix-mode-manual",
                                                                                n_clicks=0,
                                                                                size="sm",
                                                                                color="secondary",
                                                                                outline=True,
                                                                                style={"marginLeft": "0.3rem"},
                                                                            ),
                                                                        ],
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "alignItems": "center",
                                                                    "marginBottom": "0.35rem",
                                                                    "gap": "0.4rem",
                                                                },
                                                            ),
                                                            
                                                            # --- CORR5 manual VIX sliders (hidden in AUTO) ---
                                                            html.Div(
                                                                id="p2-corr5-vix-manual-container",
                                                                style={
                                                                    "display": "none",   # shown only in MANUAL
                                                                    "marginBottom": "0.4rem",
                                                                },
                                                                children=[
                                                                    html.Div(
                                                                        "Manual VIX boundaries (three cut levels, low → high):",
                                                                        style={
                                                                            "fontSize": "0.75rem",
                                                                            "color": "#CCCCCC",
                                                                            "marginBottom": "0.15rem",
                                                                        },
                                                                    ),
                                                                    dcc.Slider(
                                                                        id="p2-corr5-vix-slider-1",
                                                                        min=VIX_SLIDER_MIN,
                                                                        max=VIX_SLIDER_MAX,
                                                                        step=1,
                                                                        value=15,
                                                                        marks=VIX_MARKS,
                                                                        tooltip={"always_visible": False},
                                                                    ),
                                                                    dcc.Slider(
                                                                        id="p2-corr5-vix-slider-2",
                                                                        min=VIX_SLIDER_MIN,
                                                                        max=VIX_SLIDER_MAX,
                                                                        step=1,
                                                                        value=20,
                                                                        marks=VIX_MARKS,
                                                                        tooltip={"always_visible": False},
                                                                    ),
                                                                    dcc.Slider(
                                                                        id="p2-corr5-vix-slider-3",
                                                                        min=VIX_SLIDER_MIN,
                                                                        max=VIX_SLIDER_MAX,
                                                                        step=1,
                                                                        value=30,
                                                                        marks=VIX_MARKS,
                                                                        tooltip={"always_visible": False},
                                                                    ),
                                                                ],
                                                            ),
                                                            dcc.Graph(
                                                                id="p2-corr5-vix-heatmap",
                                                                figure={},
                                                                style={"height": "300px"},
                                                            ),
                                                        ]
                                                    ),
                                                    md=12,
                                                )
                                            ],
                                            className="mb-3",
                                        ),
                                        # ---------------- Row 4: CORR6 – Drawdown overlap matrix -------------------
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "CORR6 – Drawdown overlap matrix",
                                                                        style={
                                                                            "fontSize": "0.8rem",
                                                                            "fontWeight": "bold",
                                                                            "color": "#DDDDDD",
                                                                        },
                                                                    ),
                                                                    html.Span(
                                                                        " ⓘ",
                                                                        title=(
                                                                            "For each pair of strategies, fraction of days where both are in drawdown "
                                                                            "(dd > 0) relative to days where at least one is in drawdown. "
                                                                            "1 = always in drawdown together, 0 = never."
                                                                        ),
                                                                        style={
                                                                            "marginLeft": "0.35rem",
                                                                            "cursor": "help",
                                                                            "fontSize": "0.8rem",
                                                                            "color": "#AAAAAA",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "alignItems": "center",
                                                                    "marginBottom": "0.25rem",
                                                                },
                                                            ),
                                                            dcc.Graph(
                                                                id="p2-corr-ddoverlap-heatmap",
                                                                figure={},
                                                                style={"height": "300px"},
                                                            ),
                                                        ]
                                                    ),
                                                    md=12,
                                                )
                                            ],
                                            className="mb-3",
                                        ),
                                        #here
                                        # ---------------- Row 5: CORR7 – Drawdown depth correlation -------------------
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "CORR7 – Drawdown depth correlation",
                                                                        style={
                                                                            "fontSize": "0.8rem",
                                                                            "fontWeight": "bold",
                                                                            "color": "#DDDDDD",
                                                                        },
                                                                    ),
                                                                    html.Span(
                                                                        " ⓘ",
                                                                        title=(
                                                                            "Correlation of drawdown depths between strategies, "
                                                                            "computed only on days when both are in drawdown (dd > 0)."
                                                                        ),
                                                                        style={
                                                                            "marginLeft": "0.35rem",
                                                                            "cursor": "help",
                                                                            "fontSize": "0.8rem",
                                                                            "color": "#AAAAAA",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "alignItems": "center",
                                                                    "marginBottom": "0.25rem",
                                                                },
                                                            ),
                                                            dcc.Graph(
                                                                id="p2-corr-dddepth-heatmap",
                                                                figure={},
                                                                style={"height": "300px"},
                                                            ),
                                                        ]
                                                    ),
                                                    md=12,
                                                )
                                            ],
                                            className="mb-3",
                                        ),
                                        #here
                                        # ---------------- Row 6: BETA1 – Beta vs portfolio -------------------
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "BETA1 – Strategy beta vs portfolio",
                                                                        style={
                                                                            "fontSize": "0.8rem",
                                                                            "fontWeight": "bold",
                                                                            "color": "#DDDDDD",
                                                                        },
                                                                    ),
                                                                    html.Span(
                                                                        " ⓘ",
                                                                        title=(
                                                                            "For each strategy, OLS beta of daily P&L versus the portfolio daily P&L. "
                                                                            "Beta > 1 means the strategy tends to amplify portfolio moves; beta < 0 "
                                                                            "means it tends to move opposite to the portfolio."
                                                                        ),
                                                                        style={
                                                                            "marginLeft": "0.35rem",
                                                                            "cursor": "help",
                                                                            "fontSize": "0.8rem",
                                                                            "color": "#AAAAAA",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "alignItems": "center",
                                                                    "marginBottom": "0.25rem",
                                                                },
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        dcc.Graph(
                                                                            id="p2-beta-portfolio-bar",
                                                                            figure={},
                                                                            style={"height": "280px"},
                                                                        ),
                                                                        md=7,
                                                                    ),
                                                                    dbc.Col(
                                                                        html.Div(
                                                                            id="p2-beta-summary-table",
                                                                            style={
                                                                                "fontSize": "0.75rem",
                                                                                "color": "#DDDDDD",
                                                                                "overflowY": "auto",
                                                                                "maxHeight": "280px",
                                                                            },
                                                                        ),
                                                                        md=5,
                                                                    ),
                                                                ]
                                                            ),
                                                        ]
                                                    ),
                                                    md=12,
                                                )
                                            ],
                                            className="mb-3",
                                        ),
                                        #here
                                        # ---------------- Row 7: SD1 – Serial dependence (lag-1 autocorrelation) -------------------
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "SD1 – Serial dependence (lag-1 autocorrelation)",
                                                                        style={
                                                                            "fontSize": "0.8rem",
                                                                            "fontWeight": "bold",
                                                                            "color": "#DDDDDD",
                                                                        },
                                                                    ),
                                                                    html.Span(
                                                                        " ⓘ",
                                                                        title=(
                                                                            "For each strategy, lag-1 autocorrelation of daily P&L. "
                                                                            "Values near 0 indicate no serial dependence; positive values "
                                                                            "indicate clustering of gains/losses; negative values indicate "
                                                                            "mean-reversion."
                                                                        ),
                                                                        style={
                                                                            "marginLeft": "0.35rem",
                                                                            "cursor": "help",
                                                                            "fontSize": "0.8rem",
                                                                            "color": "#AAAAAA",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "alignItems": "center",
                                                                    "marginBottom": "0.25rem",
                                                                },
                                                            ),
                                                            dcc.Graph(
                                                                id="p2-serial-ac-bar",
                                                                figure={},
                                                                style={"height": "260px"},
                                                            ),
                                                        ]
                                                    ),
                                                    md=12,
                                                )
                                            ],
                                            className="mb-3",
                                        ),



                                    ],
                                    style={"padding": "0.75rem", "fontSize": "0.85rem"},
                                ),
                            ),
                            # ------------------------------------------------------------------
                            # TAB – Portfolio Robustness (new)
                            # ------------------------------------------------------------------
                            dcc.Tab(
                                label="Robustness",
                                value="p2-robustness",
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
                                children=html.Div(
                                    [
                                        # Short explanation
                                        html.Div(
                                            [
                                                html.Span(
                                                    "Robustness analysis on portfolio daily P&L (current factors / weights).",
                                                    style={
                                                        "fontSize": "0.8rem",
                                                        "color": "#DDDDDD",
                                                    },
                                                ),
                                                html.Br(),
                                                html.Span(
                                                    "Runs block bootstrap and Monte Carlo on portfolio daily returns, "
                                                    "similar to Phase 1 Overfitting but at portfolio level.",
                                                    style={
                                                        "fontSize": "0.75rem",
                                                        "color": "#AAAAAA",
                                                    },
                                                ),
                                            ],
                                            style={"marginBottom": "0.75rem"},
                                        ),
                            
                                        # ------------------------------------------------------------------
                                        # Controls row – Bootstrap (left) and MC (right)
                                        # ------------------------------------------------------------------
                                        dbc.Row(
                                            [
                                                # BOOTSTRAP CONTROLS
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "BOOT – Block bootstrap on daily returns",
                                                                        style={
                                                                            "fontSize": "0.8rem",
                                                                            "fontWeight": "bold",
                                                                            "color": "#DDDDDD",
                                                                        },
                                                                    ),
                                                                    html.Span(
                                                                        " ⓘ",
                                                                        title=(
                                                                            "Resamples blocks of consecutive daily portfolio returns "
                                                                            "to preserve streaks. Produces a distribution of Sharpe, "
                                                                            "Max DD and final return."
                                                                        ),
                                                                        style={
                                                                            "marginLeft": "0.35rem",
                                                                            "cursor": "help",
                                                                            "fontSize": "0.8rem",
                                                                            "color": "#AAAAAA",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "alignItems": "center",
                                                                    "marginBottom": "0.35rem",
                                                                },
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Bootstrap runs",
                                                                                style={
                                                                                    "fontSize": "0.75rem",
                                                                                    "color": "#CCCCCC",
                                                                                },
                                                                            ),
                                                                            dcc.Input(
                                                                                id="p2-robust-boot-n-sim",
                                                                                type="number",
                                                                                min=100,
                                                                                step=100,
                                                                                value=2000,
                                                                                style={
                                                                                    "width": "70px",
                                                                                    "fontSize": "0.75rem",
                                                                                    "backgroundColor": "#2a2a2a",
                                                                                    "color": "#EEEEEE",
                                                                                    "border": "1px solid #555555",
                                                                                },
                                                                            ),
                                                                        ],
                                                                        md=3,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Block length (days)",
                                                                                style={
                                                                                    "fontSize": "0.75rem",
                                                                                    "color": "#CCCCCC",
                                                                                },
                                                                            ),
                                                                            dcc.Input(
                                                                                id="p2-robust-boot-block-len",
                                                                                type="number",
                                                                                min=1,
                                                                                step=1,
                                                                                value=5,
                                                                                style={
                                                                                    "width": "50px",
                                                                                    "fontSize": "0.75rem",
                                                                                    "backgroundColor": "#2a2a2a",
                                                                                    "color": "#EEEEEE",
                                                                                    "border": "1px solid #555555",
                                                                                },
                                                                            ),
                                                                        ],
                                                                        md=3,
                                                                    ),
                                                                    dbc.Col(
                                                                        html.Div(
                                                                            [
                                                                                html.Label(
                                                                                    "Initial equity ($)",
                                                                                    style={"fontSize": "0.8rem", "marginBottom": "0.1rem"},
                                                                                ),
                                                                                dcc.Input(
                                                                                    id="p2-initial-equity-input",
                                                                                    type="number",
                                                                                    value=100000,
                                                                                    min=0,
                                                                                    step=1000,
                                                                                    style={
                                                                                        "width": "70%",
                                                                                        "backgroundColor": "#333333",
                                                                                        "color": "#FFFFFF",
                                                                                        "border": "1px solid #555555",
                                                                                    },
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        md=3,
                                                                    ),
                                                                    dbc.Col(
                                                                        html.Div(
                                                                            [
                                                                                html.Label(
                                                                                    "Ruin threshold ($)",
                                                                                    style={"fontSize": "0.8rem", "marginBottom": "0.1rem"},
                                                                                ),
                                                                                dcc.Input(
                                                                                    id="p2-robust-ruin-threshold",
                                                                                    type="number",
                                                                                    value=100000,
                                                                                    step=1000,
                                                                                    style={
                                                                                        "width": "70%",
                                                                                        "backgroundColor": "#333333",
                                                                                        "color": "#FFFFFF",
                                                                                        "border": "1px solid #555555",
                                                                                    },
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        md=3,
                                                                    ),

                                                                ],
                                                                className="mb-2",
                                                            ),
                                                            dbc.Button(
                                                                "Run Bootstrap",
                                                                id="p2-robust-boot-run-btn",
                                                                n_clicks=0,
                                                                color="primary",
                                                                size="sm",
                                                            ),
                                                            html.Div(
                                                                id="p2-robust-boot-status",
                                                                style={
                                                                    "marginTop": "0.35rem",
                                                                    "fontSize": "0.75rem",
                                                                    "color": "#AAAAAA",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                    md=6,
                                                ),
                            
                                                # MONTE CARLO CONTROLS
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "MC – Monte Carlo on daily returns (Gaussian)",
                                                                        style={
                                                                            "fontSize": "0.8rem",
                                                                            "fontWeight": "bold",
                                                                            "color": "#DDDDDD",
                                                                        },
                                                                    ),
                                                                    html.Span(
                                                                        " ⓘ",
                                                                        title=(
                                                                            "Simulates portfolio equity paths by drawing daily returns "
                                                                            "from a Gaussian calibrated on historical portfolio returns. "
                                                                            "Used for final equity and drawdown risk distribution."
                                                                        ),
                                                                        style={
                                                                            "marginLeft": "0.35rem",
                                                                            "cursor": "help",
                                                                            "fontSize": "0.8rem",
                                                                            "color": "#AAAAAA",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "alignItems": "center",
                                                                    "marginBottom": "0.35rem",
                                                                },
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Number of MC paths",
                                                                                style={
                                                                                    "fontSize": "0.75rem",
                                                                                    "color": "#CCCCCC",
                                                                                },
                                                                            ),
                                                                            dcc.Input(
                                                                                id="p2-robust-mc-n-sim",
                                                                                type="number",
                                                                                min=100,
                                                                                step=100,
                                                                                value=5000,
                                                                                style={
                                                                                    "width": "70px",
                                                                                    "fontSize": "0.75rem",
                                                                                    "backgroundColor": "#2a2a2a",
                                                                                    "color": "#EEEEEE",
                                                                                    "border": "1px solid #555555",
                                                                                },
                                                                            ),
                                                                        ],
                                                                        md=4,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Horizon (days, 0 = use sample length)",
                                                                                style={
                                                                                    "fontSize": "0.75rem",
                                                                                    "color": "#CCCCCC",
                                                                                },
                                                                            ),
                                                                            dcc.Input(
                                                                                id="p2-robust-mc-horizon-days",
                                                                                type="number",
                                                                                min=0,
                                                                                step=10,
                                                                                value=0,
                                                                                style={
                                                                                    "width": "50px",
                                                                                    "fontSize": "0.75rem",
                                                                                    "backgroundColor": "#2a2a2a",
                                                                                    "color": "#EEEEEE",
                                                                                    "border": "1px solid #555555",
                                                                                },
                                                                            ),
                                                                        ],
                                                                        md=5,
                                                                    ),
                                                                ],
                                                                className="mb-2",
                                                            ),
                                                            dbc.Button(
                                                                "Run Monte Carlo",
                                                                id="p2-robust-mc-run-btn",
                                                                n_clicks=0,
                                                                color="primary",
                                                                size="sm",
                                                            ),
                                                            html.Div(
                                                                id="p2-robust-mc-status",
                                                                style={
                                                                    "marginTop": "0.35rem",
                                                                    "fontSize": "0.75rem",
                                                                    "color": "#AAAAAA",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                            
                                        html.Hr(
                                            style={
                                                "marginTop": "0.25rem",
                                                "marginBottom": "0.75rem",
                                            }
                                        ),
                            
                                        # ------------------------------------------------------------------
                                        # Bootstrap charts + metrics
                                        # ------------------------------------------------------------------
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-robust-boot-sharpe-hist",
                                                        figure={},
                                                        style={"height": "280px"},
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-robust-boot-maxdd-hist",
                                                        figure={},
                                                        style={"height": "280px"},
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="mb-2",
                                        ),
                                        
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-robust-boot-maxblock-hist",
                                                        figure={},
                                                        style={"height": "260px"},
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        id="p2-robust-boot-metrics-table",
                                                        style={
                                                            "fontSize": "0.8rem",
                                                            "backgroundColor": "#222222",
                                                            "border": "1px solid #444444",
                                                            "padding": "0.5rem",
                                                            "maxHeight": "260px",
                                                            "overflowY": "auto",
                                                        },
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),

                                        
                                        # NEW: ECDF charts for Ending Equity and Max DD
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-robust-boot-final-ecdf",
                                                        figure={},
                                                        style={"height": "260px"},
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-robust-boot-maxdd-ecdf",
                                                        figure={},
                                                        style={"height": "260px"},
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),

                            
                                        html.Hr(
                                            style={
                                                "marginTop": "0.25rem",
                                                "marginBottom": "0.75rem",
                                            }
                                        ),
                            
                                        # ------------------------------------------------------------------
                                        # MC charts + metrics
                                        # ------------------------------------------------------------------
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-robust-mc-final-equity-hist",
                                                        figure={},
                                                        style={"height": "280px"},
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-robust-mc-maxdd-hist",
                                                        figure={},
                                                        style={"height": "280px"},
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="mb-2",
                                        ),
                                        #here
                                        # Row 5: MC fan chart + metrics table
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-robust-mc-fan-chart",
                                                        figure={},
                                                        style={"height": "320px"},
                                                    ),
                                                    md=8,
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        id="p2-robust-mc-metrics-table",
                                                        style={
                                                            "fontSize": "0.8rem",
                                                            "backgroundColor": "#222222",
                                                            "border": "1px solid #444444",
                                                            "padding": "0.5rem",
                                                            "maxHeight": "320px",
                                                            "overflowY": "auto",
                                                        },
                                                    ),
                                                    md=4,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        
                                        # Row 6: MC ECDF final equity + probability of ruin chart
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-robust-mc-final-ecdf",
                                                        figure={},
                                                        style={"height": "260px"},
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-robust-mc-ruin-bar",
                                                        figure={},
                                                        style={"height": "260px"},
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        # ------------------------------
                                        # Row 7: Random start-date analysis
                                        # ------------------------------
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        [
                                                            dbc.CardHeader("Random start-date analysis"),
                                                            dbc.CardBody(
                                                                [
                                                                    html.Div(
                                                                        "Draw N random start dates and evaluate the next M months as a contiguous real window (no reshuffle).",
                                                                        style={"fontSize": "0.8rem", "color": "#BBBBBB", "marginBottom": "0.75rem"},
                                                                    ),
                                        
                                                                    html.Div("N periods", style={"fontSize": "0.8rem", "marginTop": "0.25rem"}),
                                                                    dcc.Input(
                                                                        id="p2-randstart-n-periods",
                                                                        type="number",
                                                                        min=5,
                                                                        step=1,
                                                                        value=50,
                                                                        style={"width": "40%"},
                                                                    ),
                                        
                                                                    html.Div("Period length (months)", style={"fontSize": "0.8rem", "marginTop": "0.6rem"}),
                                                                    dcc.Input(
                                                                        id="p2-randstart-months",
                                                                        type="number",
                                                                        min=1,
                                                                        step=1,
                                                                        value=6,
                                                                        style={"width": "40%"},
                                                                    ),
                                        
                                                                    dbc.Checklist(
                                                                        id="p2-randstart-no-overlap",
                                                                        options=[{"label": "Avoid overlapping windows (best effort)", "value": "no_overlap"}],
                                                                        value=[],
                                                                        style={"marginTop": "0.75rem", "fontSize": "0.85rem"},
                                                                    ),
                                        
                                                                    dbc.Button(
                                                                        "Run random start-date analysis",
                                                                        id="p2-randstart-run-btn",
                                                                        n_clicks=0,
                                                                        color="primary",
                                                                        className="mt-2",
                                                                        style={"width": "100%"},
                                                                    ),
                                        
                                                                    html.Div(
                                                                        id="p2-randstart-status",
                                                                        style={"marginTop": "0.6rem", "fontSize": "0.8rem", "color": "#BBBBBB"},
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        style={"backgroundColor": "#222222", "border": "1px solid #333333"},
                                                    ),
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        [
                                                            dbc.CardBody(
                                                                [
                                                                    dcc.Graph(
                                                                        id="p2-randstart-fig",
                                                                        figure={},
                                                                        config={"displayModeBar": True, "displaylogo": False},
                                                                        style={"height": "420px"},
                                                                    )
                                                                ]
                                                            )
                                                        ],
                                                        style={"backgroundColor": "#222222", "border": "1px solid #333333"},
                                                    ),
                                                    width=9,
                                                ),
                                            ],
                                            className="g-2",
                                            style={"marginTop": "0.75rem"},
                                        ),
                                        # Row 8: Random start-date – dispersion view
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="p2-randstart-dist-fig",
                                                        figure={},
                                                        style={"height": "260px"},
                                                    ),
                                                    md=7,
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        id="p2-randstart-metrics",
                                                        style={
                                                            "fontSize": "0.8rem",
                                                            "backgroundColor": "#222222",
                                                            "border": "1px solid #444444",
                                                            "padding": "0.5rem",
                                                            "maxHeight": "260px",
                                                            "overflowY": "auto",
                                                        },
                                                    ),
                                                    md=5,
                                                ),
                                            ],
                                            className="mb-3",
                                            style={"marginTop": "0.5rem"},
                                        ),

                                    ],
                                    style={"padding": "0.75rem", "fontSize": "0.85rem"},
                                ),
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

#--- CORR5 VIX regime AUTO/MANUAL

@callback(
    Output("p2-corr5-vix-manual-container", "style"),
    Output("p2-corr5-vix-mode-auto", "outline"),
    Output("p2-corr5-vix-mode-manual", "outline"),
    Input("p2-corr5-vix-mode-auto", "n_clicks"),
    Input("p2-corr5-vix-mode-manual", "n_clicks"),
)
def toggle_corr5_vix_mode(n_auto, n_manual):
    """
    Simple UI toggle for CORR5 VIX regime bucketing:
    - AUTO: manual sliders hidden
    - MANUAL: manual sliders shown
    """
    n_auto = n_auto or 0
    n_manual = n_manual or 0

    # Default: AUTO if no clicks yet, otherwise whichever button was clicked more
    manual_mode = n_manual > n_auto

    if manual_mode:
        container_style = {"display": "block", "marginBottom": "0.4rem"}
        return container_style, True, False  # AUTO outlined, MANUAL solid
    else:
        container_style = {"display": "none"}
        return container_style, False, True  # AUTO solid, MANUAL outlined


# ---------------------------------------------------------------------------
# Phase 2 – internal helpers for portfolio analytics
# ---------------------------------------------------------------------------


def _metric_cell(label: str, value: str) -> html.Div:
    """
    Small metric cell styled similarly to Phase-1 header metrics.
    """
    return html.Div(
        [
            html.Div(label, style={"fontSize": "0.75rem", "color": "#AAAAAA"}),
            html.Div(
                value,
                style={
                    "fontSize": "0.95rem",
                    "fontWeight": "bold",
                    "marginTop": "0.15rem",
                },
            ),
        ],
        style={
            "padding": "0.35rem 0.6rem",
            "borderRadius": "4px",
            "backgroundColor": "#2b2b2b",
            "border": "1px solid #444444",
            "marginRight": "0.5rem",
            "marginBottom": "0.4rem",
            "minWidth": "120px",
        },
    )


def _losing_streak_stats(daily_pnl: pd.Series) -> tuple[int, float]:
    """
    Compute worst and average losing streak (consecutive negative days).
    Only streaks with length >= 2 are counted for the average.
    """
    max_streak = 0
    streaks = []

    current = 0
    for v in daily_pnl:
        if v < 0:
            current += 1
        else:
            if current >= 2:
                streaks.append(current)
                max_streak = max(max_streak, current)
            current = 0

    # handle trailing streak
    if current >= 2:
        streaks.append(current)
        max_streak = max(max_streak, current)

    if streaks:
        avg_streak = float(np.mean(streaks))
    else:
        avg_streak = 0.0
        max_streak = 0

    return max_streak, avg_streak


def _build_portfolio_timeseries(
    active_store: list,
    weights_store: dict | None,
    weight_mode: str,
    initial_equity: float,
) -> dict | None:
    """
    Core engine: take selected strategies from Active list, their df's from
    p1_strategy_store, and compute portfolio daily P&L, equity and DD.

    - P&L date = 'Date Closed' (summed by day)
    - weight_mode: 'factors' or 'lots'
    - factors = size multipliers (fractional lots)
    - lots_vec = integer lots in 'lots' mode
    - P&L is always ADDITIVE: Σ lots_i * P&L_i (no normalisation)
    """

    active_store = active_store or []

    # Use only SELECTED strategies in Active list
    selected_rows = [r for r in active_store if r.get("is_selected")]
    if not selected_rows:
        return None

    # Build per-strategy daily P&L series
    pnl_series_by_uid: dict[str, pd.Series] = {}

    for row in selected_rows:
        uid = row.get("uid")
        if not uid:
            continue

        meta = p1_strategy_store.get(uid, {})
        df = meta.get("df")
        if df is None:
            continue

        if "Date Closed" not in df.columns or "P/L" not in df.columns:
            continue

        tmp = df[["Date Closed", "P/L"]].copy()
        tmp["Date Closed"] = pd.to_datetime(tmp["Date Closed"]).dt.date
        s = tmp.groupby("Date Closed")["P/L"].sum()

        if not s.empty:
            pnl_series_by_uid[uid] = s

    if not pnl_series_by_uid:
        return None

    # Align all series onto a common daily index
    all_dates = sorted(set().union(*[s.index for s in pnl_series_by_uid.values()]))
    idx = pd.to_datetime(all_dates)

    pnl_df = pd.DataFrame(index=idx)
    for uid, s in pnl_series_by_uid.items():
        pnl_df[uid] = s.reindex(all_dates, fill_value=0.0).values

    active_uids = list(pnl_df.columns)
    n_strats = len(active_uids)

    # ---------------- factors -> lots_vec (size) ----------------
    weights_store = weights_store or {}

    # base factors from store (>=0, fallback 1.0)
    factors = []
    for uid in active_uids:
        entry = weights_store.get(uid, {}) if isinstance(weights_store, dict) else {}
        f = entry.get("factor", 1.0)
        try:
            f = float(f)
        except (TypeError, ValueError):
            f = 1.0
        if f < 0:
            f = 0.0
        factors.append(f)

    factors = np.array(factors, dtype=float)

    # lots_vec is what actually scales P&L
    if weight_mode == "lots":
        # Integer lots preview:
        #  - factor <= 0  -> 0 lots
        #  - factor > 0   -> at least 1 lot (round, then floor to 1)
        lots_vec = np.zeros_like(factors, dtype=int)
        positive_mask = factors > 0
        if positive_mask.any():
            rounded = np.round(factors[positive_mask]).astype(int)
            rounded[rounded < 1] = 1
            lots_vec[positive_mask] = rounded
        # fallback if everything is 0 for some reason
        if lots_vec.sum() == 0 and positive_mask.any():
            lots_vec[positive_mask] = 1
    else:
        # 'factors' mode: use fractional factors directly as size multipliers
        lots_vec = factors.copy()

    # Derived weights (for display/allocation only)
    total_size = float(lots_vec.sum())
    if total_size > 0:
        weights = pd.Series(lots_vec / total_size, index=active_uids)
    else:
        weights = pd.Series(
            np.ones(n_strats) / max(n_strats, 1),
            index=active_uids,
        )

    # ---------------- portfolio series (ADDITIVE) ----------------
    portfolio_daily = pd.Series(
        pnl_df.values @ lots_vec,
        index=pnl_df.index,
        name="portfolio_daily_pnl",
    )

    initial_equity = float(initial_equity) if initial_equity is not None else 100000.0
    equity = pd.Series(initial_equity + portfolio_daily.cumsum(), index=pnl_df.index)
    running_max = equity.cummax()
    dd = running_max - equity
    dd_pct = dd / running_max.replace(0, np.nan)

    return {
        "dates": pnl_df.index,
        "pnl_df": pnl_df,
        "portfolio_daily": portfolio_daily,
        "equity": equity,
        "dd": dd,
        "dd_pct": dd_pct,
        "weights": weights,      # allocation %
        "lots_vec": lots_vec,    # actual size multipliers used
        "initial_equity": initial_equity,
    }



def _compute_pnl_contribution_from_series(series: dict) -> pd.Series:
    """
    From the _build_portfolio_timeseries output, compute weighted total P&L
    contribution per strategy (absolute dollars).
    """
    if not series:
        return pd.Series(dtype=float)

    pnl_df: pd.DataFrame | None = series.get("pnl_df")
    lots_vec = series.get("lots_vec")

    if pnl_df is None or pnl_df.empty or lots_vec is None:
        return pd.Series(dtype=float)

    active_uids = list(pnl_df.columns)
    lots = pd.Series(lots_vec, index=active_uids, dtype=float)

    # Weighted daily P&L per strategy
    weighted = pnl_df.mul(lots, axis=1)

    # Absolute P&L contribution per strategy
    return weighted.sum(axis=0)


def _compute_dd_contribution_from_series(series: dict) -> pd.Series:
    """
    From the _build_portfolio_timeseries output, compute DD contribution per
    strategy: sum of weighted P&L ONLY on days where the PORTFOLIO drawdown
    deepens (dd diff > 0).
    """
    if not series:
        return pd.Series(dtype=float)

    pnl_df: pd.DataFrame | None = series.get("pnl_df")
    lots_vec = series.get("lots_vec")
    dd: pd.Series | None = series.get("dd")

    if pnl_df is None or pnl_df.empty or lots_vec is None or dd is None or dd.empty:
        return pd.Series(dtype=float)

    active_uids = list(pnl_df.columns)
    lots = pd.Series(lots_vec, index=active_uids, dtype=float)

    # Weighted daily P&L per strategy
    weighted = pnl_df.mul(lots, axis=1)

    # Identify portfolio DD-worsening days
    dd_delta = dd.diff().fillna(0.0)
    dd_worsening = dd_delta > 0

    if not dd_worsening.any():
        # No DD worsening days – everything neutral
        return pd.Series(0.0, index=active_uids, dtype=float)

    # Sum contributions only on days where portfolio DD increases
    dd_contrib = weighted[dd_worsening].sum(axis=0)

    return dd_contrib


# ---------------------------------------------------------------------------
# Phase 2 – Portfolio Analytics callback
# ---------------------------------------------------------------------------


@callback(
    Output("p2-portfolio-metrics", "children"),
    Output("p2-equity-graph", "figure"),
    Output("p2-dd-graph", "figure"),
    Output("p2-dist-metrics", "children"),
    Output("p2-pnl-histogram", "figure"),
    Output("p2-dow-bar-graph", "figure"),
    Input("p1-active-list-store", "data"),
    Input("p2-weights-store", "data"),
    #Input("p2-weight-mode", "value"),
    Input("p2-initial-equity-input", "value"),
    Input("p2-show-strategy-equity", "value"),
)
def update_portfolio_analytics(
    active_store,
    weights_store,
    #weight_mode,
    initial_equity,
    show_strategy_equity,
):
    """
    Build portfolio-level metrics & charts.

    - Metrics and primary lines always use *factors* sizing.
    - Secondary (overlay) lines for equity/DD always use *integer lots*
      as a preview.
    """
    # ------------------------------------------------------------------
    # Fixed weighting modes: main = factors, overlay = lots
    # ------------------------------------------------------------------
    main_mode = "factors"
    overlay_mode = "lots"

    base_initial_equity = float(initial_equity or 100000.0)


    # ------------------------------------------------------------------
    # Build main series (for metrics and primary lines)
    # ------------------------------------------------------------------
    series_main = _build_portfolio_timeseries(
        active_store=active_store,
        weights_store=weights_store,
        weight_mode=main_mode,
        initial_equity=base_initial_equity,
    )

    if series_main is None:
        empty_fig = {
            "data": [],
            "layout": {
                "template": "plotly_dark",
                "paper_bgcolor": "#222222",
                "plot_bgcolor": "#222222",
                "font": {"color": "#EEEEEE"},
            },
        }
        msg = html.Div(
            "No selected strategies with data – select strategies in Active list.",
            style={"fontSize": "0.8rem", "color": "#AAAAAA"},
        )
        return msg, empty_fig, empty_fig, msg, empty_fig, empty_fig

    # Try to build overlay series; if anything goes wrong, just skip overlay
    try:
        series_overlay = _build_portfolio_timeseries(
            active_store=active_store,
            weights_store=weights_store,
            weight_mode=overlay_mode,
            initial_equity=base_initial_equity,
        )
    except Exception:
        series_overlay = None

    # Unpack main series
    dates = series_main["dates"]                      # DatetimeIndex
    portfolio_daily = series_main["portfolio_daily"]
    equity_main = series_main["equity"]
    dd_main = series_main["dd"]
    dd_pct = series_main["dd_pct"]
    initial_equity_main = float(series_main["initial_equity"])

    # Unpack overlay series if available
    equity_overlay = None
    dd_overlay = None
    if series_overlay is not None:
        equity_overlay = series_overlay.get("equity")
        dd_overlay = series_overlay.get("dd")

    # ------------------------------------------------------------------
    # Metrics summary (based on main_mode)
    # ------------------------------------------------------------------
    n_days = len(portfolio_daily)
    if n_days == 0:
        total_pnl = max_dd_abs = max_dd_pct = 0.0
        sharpe_ann = cagr = mar = win_rate = 0.0
        avg_daily_pnl = avg_monthly_pnl = 0.0
        worst_ls, avg_ls = 0, 0.0
        top5_pct = float("nan")
        gini = float("nan")
        hhi = float("nan")
        pcr = float("nan")
    else:
        total_pnl = float(equity_main.iloc[-1] - initial_equity_main)
        max_dd_abs = float(dd_main.max())
        max_dd_pct = float((dd_pct.max() or 0.0) * 100.0)

        # Daily return based on initial equity, same as Phase 1 logic
        daily_ret = portfolio_daily / initial_equity_main
        mu = daily_ret.mean()
        sigma = daily_ret.std(ddof=1)
        sharpe_ann = float(mu / sigma * np.sqrt(252.0)) if sigma > 0 else 0.0

        # Years based on calendar span between first and last date
        if len(dates) > 1:
            years = max((dates[-1] - dates[0]).days / 365.25, 1e-6)
        else:
            years = max(n_days / 252.0, 1e-6)

        cagr = float((equity_main.iloc[-1] / initial_equity_main) ** (1.0 / years) - 1.0)
        mar = float(cagr / (abs(max_dd_pct) / 100.0)) if max_dd_pct > 0 else 0.0

        win_rate = float((portfolio_daily > 0).mean() * 100.0)
        avg_daily_pnl = float(portfolio_daily.mean())

        monthly_pnl = portfolio_daily.resample("ME").sum()
        avg_monthly_pnl = float(monthly_pnl.mean()) if not monthly_pnl.empty else 0.0

        worst_ls, avg_ls = _losing_streak_stats(portfolio_daily)

        # Concentration: % of total P&L coming from top 5% days
        if n_days >= 5 and abs(total_pnl) > 0:
            q = int(max(1, np.floor(n_days * 0.05)))
            top5_sum = float(portfolio_daily.nlargest(q).sum())
            top5_pct = float(top5_sum / total_pnl * 100.0)
        else:
            top5_pct = float("nan")

        # Gini & HHI of positive-day contributions (wins only)
        pos = portfolio_daily[portfolio_daily > 0]
        if len(pos) >= 2 and pos.sum() > 0:
            shares = (pos / pos.sum()).values  # each win's fraction of total wins P&L

            # HHI: sum of squared shares
            hhi = float(np.sum(shares ** 2))

            # Gini: standard discrete formula on shares
            shares_sorted = np.sort(shares)
            n_pos = len(shares_sorted)
            index = np.arange(1, n_pos + 1)
            gini = float(
                np.sum((2 * index - n_pos - 1) * shares_sorted)
                / (n_pos * np.sum(shares_sorted))
            )
        else:
            gini = float("nan")
            hhi = float("nan")

        # ------------------- Premium Capture Rate (PCR) --------------------
        total_premium_abs = 0.0

        if active_store:
            for row in active_store:
                # Only selected strategies contribute to portfolio & PCR
                if not row.get("is_selected", False):
                    continue
                sid = row.get("sid")
                uid = row.get("uid")

                meta = None
                if sid and sid in p1_strategy_store:
                    meta = p1_strategy_store[sid]
                elif uid and uid in p1_strategy_store:
                    meta = p1_strategy_store[uid]

                if not meta:
                    continue

                df = meta.get("df")
                if df is None or df.empty:
                    continue

                prem_col = None
                for c in df.columns:
                    cl = str(c).lower()
                    if "premium" in cl or "prem" in cl:
                        prem_col = c
                        break

                if prem_col is None:
                    continue

                total_premium_abs += float(df[prem_col].abs().sum())

        if total_premium_abs > 0.0:
            pcr = float(total_pnl / total_premium_abs)
        else:
            pcr = float("nan")

    metrics_bar = html.Div(
        [
            _metric_cell("Total P&L", f"${total_pnl:,.0f}"),
            _metric_cell("Max DD ($)", f"${max_dd_abs:,.0f}"),
            _metric_cell("Max DD (%)", f"{max_dd_pct:,.2f}%"),
            _metric_cell("CAGR", f"{cagr * 100:,.2f}%"),
            _metric_cell("MAR", f"{mar:,.2f}"),
            _metric_cell("Sharpe (ann.)", f"{sharpe_ann:,.2f}"),
            _metric_cell(
                "PCR (P&L / |prem|)",
                "N/A" if np.isnan(pcr) else f"{pcr * 100:,.2f}%",
            ),
            _metric_cell("Avg daily P&L", f"${avg_daily_pnl:,.0f}"),
            _metric_cell("Avg monthly P&L", f"${avg_monthly_pnl:,.0f}"),
            _metric_cell("Win rate (days)", f"{win_rate:,.1f}%"),
            _metric_cell("Worst losing streak", f"{worst_ls} days"),
            _metric_cell(
                "Avg losing streak (>=2)", f"{avg_ls:,.1f} days"
            ),
            _metric_cell(
                "Top-5% days P&L",
                "N/A" if np.isnan(top5_pct) else f"{top5_pct:,.1f}%",
            ),
            _metric_cell(
                "Gini (wins)",
                "N/A" if np.isnan(gini) else f"{gini:,.2f}",
            ),
            _metric_cell(
                "HHI (wins)",
                "N/A" if np.isnan(hhi) else f"{hhi:,.3f}",
            ),
        ],
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "alignItems": "flex-start",
        },
    )
    
    # ------------------------------------------------------------------
    # Equity and DD figures (main + overlay)
    # ------------------------------------------------------------------
    def _mode_label(mode: str) -> str:
        if mode == "lots":
            return "Integer lots"
        if mode == "equal":
            return "Equal weights (1/N)"
        return "Factors"

   
    equity_fig = go.Figure()

    # Primary line (based on main_mode)
    label_main = f"Portfolio equity – {_mode_label(main_mode)}"
    equity_fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity_main,
            mode="lines",
            name=label_main,
            hovertemplate=(
            f"{label_main}<br>$%{{y:,.0f}}<br>%{{x|%d-%b-%y}}<extra></extra>"
            ),
        )
    )


    # Overlay line if available (based on overlay_mode)
    
    if equity_overlay is not None:
        try:
            label_overlay = f"Portfolio equity – {_mode_label(overlay_mode)}"
            equity_fig.add_trace(
                go.Scatter(
                    x=series_overlay["dates"],
                    y=equity_overlay,
                    mode="lines",
                    name=label_overlay,
                    line={"dash": "dot"},
                    hovertemplate=(
                    f"{label_main}<br>$%{{y:,.0f}}<br>%{{x|%d-%b-%y}}<extra></extra>"
                    ),
                )
            )
        except Exception:
            pass


    equity_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin=dict(l=40, r=10, t=35, b=40),
        xaxis_title="Date",
        yaxis_title="Equity",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )

    # ------------------------------------------------------------------
    # Optional overlay: individual strategy equity curves
    # ------------------------------------------------------------------
    show_strategy_equity = bool(show_strategy_equity)

    if show_strategy_equity and active_store:
        active_store = active_store or []

        for row in active_store:
            # Only overlay selected strategies
            if not row.get("is_selected", False):
                continue
            sid = row.get("sid")
            uid = row.get("uid")
            name = row.get("name") or uid or sid

            meta = None
            if sid and sid in p1_strategy_store:
                meta = p1_strategy_store[sid]
            elif uid and uid in p1_strategy_store:
                meta = p1_strategy_store[uid]

            if not meta:
                continue

            df = meta.get("df")
            if df is None or df.empty:
                continue

            if "Date Closed" not in df.columns or "P/L" not in df.columns:
                continue

            tmp = df[["Date Closed", "P/L"]].copy()
            # Normalise exactly as in _build_portfolio_timeseries
            tmp["Date Closed"] = pd.to_datetime(tmp["Date Closed"]).dt.date

            daily = (
                tmp.groupby("Date Closed")["P/L"]
                .sum()
                .sort_index()
            )
            if daily.empty:
                continue

            daily.index = pd.to_datetime(daily.index)
            daily = daily.reindex(dates, fill_value=0.0)

            strat_equity = initial_equity_main + daily.cumsum()
            
            display_name = f"{name} (1x)"

            equity_fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=strat_equity,
                    mode="lines",
                    line=dict(
                        width=1,
                        dash="dot",
                        color=strategy_color_for_uid(uid),
                    ),
                    opacity=0.5,
                    name=display_name,
                    showlegend=True,
                    hovertemplate=(
                        f"{name}<br>$%{{y:,.0f}}<br>%{{x|%d-%b-%y}}"
                        "<extra></extra>"
                    ),
                )
            )


    # ------------------------------------------------------------------
    # Drawdown figure
    # ------------------------------------------------------------------
    dd_fig = go.Figure()

    dd_fig.add_trace(
        go.Scatter(
            x=dates,
            y=-dd_main,
            mode="lines",
            name=f"Drawdown – {_mode_label(main_mode)}",
        )
    )

    if dd_overlay is not None:
        try:
            dd_fig.add_trace(
                go.Scatter(
                    x=series_overlay["dates"],
                    y=-dd_overlay,
                    mode="lines",
                    name=f"Drawdown – {_mode_label(overlay_mode)}",
                    line={"dash": "dot"},
                )
            )
        except Exception:
            pass

    dd_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin=dict(l=40, r=10, t=35, b=40),
        xaxis_title="Date",
        yaxis_title="Drawdown",
        legend=dict(
            orientation="h",
            yanchor="middle",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )

    # ------------------------------------------------------------------
    # Distribution metrics + histogram (based on main series)
    # ------------------------------------------------------------------
    skew = float(portfolio_daily.skew()) if n_days > 1 else 0.0
    kurt = float(portfolio_daily.kurtosis()) if n_days > 1 else 0.0

    if n_days > 10:
        q = int(max(1, np.floor(n_days * 0.05)))
        top = portfolio_daily.nlargest(q).mean()
        bottom = portfolio_daily.nsmallest(q).mean()
        tail_ratio = float(top / abs(bottom)) if bottom < 0 else np.nan
    else:
        tail_ratio = np.nan

    dist_metrics = html.Div(
        [
            html.Span(f"Skewness: {skew:,.2f}", style={"marginRight": "1rem"}),
            html.Span(f"Kurtosis: {kurt:,.2f}", style={"marginRight": "1rem"}),
            html.Span(
                "Tail ratio: N/A"
                if np.isnan(tail_ratio)
                else f"Tail ratio: {tail_ratio:,.2f}",
                style={"marginRight": "1rem"},
            ),
            html.Span(
                "Top-5% days P&L: N/A"
                if np.isnan(top5_pct)
                else f"Top-5% days P&L: {top5_pct:,.1f}%",
            ),
        ],
        style={"fontSize": "0.8rem"},
    )

    hist_fig = go.Figure()
    hist_fig.add_trace(
        go.Histogram(
            x=portfolio_daily,
            nbinsx=40,
            name="Daily P&L",
        )
    )
    hist_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin=dict(l=40, r=10, t=25, b=40),
        xaxis_title="Daily P&L",
        yaxis_title="Count",
        showlegend=False,
    )

    # ------------------------------------------------------------------
    # Day-of-week exposure bar (based on main series)
    # ------------------------------------------------------------------
    dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    dow = portfolio_daily.copy()
    dow.index = pd.DatetimeIndex(dow.index)
    dow_group = dow.groupby(dow.index.dayofweek).sum()

    dow_x = [dow_map.get(i, str(i)) for i in dow_group.index]
    dow_y = [float(v) for v in dow_group.values]

    dow_fig = go.Figure()
    dow_fig.add_trace(
        go.Bar(
            x=dow_x,
            y=dow_y,
            name="Total P&L by weekday",
        )
    )
    dow_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin=dict(l=40, r=10, t=25, b=40),
        xaxis_title="Day of week",
        yaxis_title="Total P&L",
        showlegend=False,
    )

    return metrics_bar, equity_fig, dd_fig, dist_metrics, hist_fig, dow_fig

#--------- Portfolio COntribution P&L and DD chart
@callback(
    Output("p2-contr-pnl-waterfall", "figure"),
    Output("p2-contr-dd-bars", "figure"),
    Input("p1-active-list-store", "data"),
    Input("p2-weights-store", "data"),
    Input("p2-initial-equity-input", "value"),
    Input("p2-contr-pnl-mode", "value"),
)
def update_portfolio_contribution_charts(
    active_store,
    weights_store,
    initial_equity,
    pnl_mode,
):
    """
    Build:
      - P&L contribution waterfall (ABS / %)
      - DD contribution horizontal bar chart
    using the same portfolio engine as the main Analytics callback.
    """

    # Normalise inputs
    active_store = active_store or []
    weights_store = weights_store or {}
    base_initial_equity = float(initial_equity or 100000.0)
    pnl_mode = pnl_mode or "abs"

    # Reuse the core engine (always 'factors' mode for Analytics)
    series = _build_portfolio_timeseries(
        active_store=active_store,
        weights_store=weights_store,
        weight_mode="factors",
        initial_equity=base_initial_equity,
    )

    if series is None:
        empty = _empty_corr_figure("No selected strategies.")
        return empty, empty

    pnl_df: pd.DataFrame = series["pnl_df"]
    if pnl_df.empty or pnl_df.shape[1] < 1:
        empty = _empty_corr_figure("No P&L data available.")
        return empty, empty

    # ---------- P&L contribution ----------
    contr_abs = _compute_pnl_contribution_from_series(series)

    # If all zero, avoid division by zero in % mode
    if pnl_mode == "pct":
        total = float(contr_abs.sum())
        if abs(total) > 0:
            contr_vals = contr_abs / total * 100.0
        else:
            contr_vals = contr_abs * 0.0
        y_title_pnl = "% of total portfolio P&L"
    else:
        contr_vals = contr_abs
        y_title_pnl = "Contribution to total P&L ($)"

    # Labels: use strategy names from Active list, aligned to pnl_df columns
    uid_to_name: dict[str, str] = {}
    for row in active_store:
        if not row.get("is_selected", False):
            continue
        uid = row.get("uid")
        if not uid:
            continue
        raw_name = row.get("name") or uid or row.get("sid")
        uid_to_name[uid] = raw_name

    ordered_uids = [uid for uid in pnl_df.columns if uid in uid_to_name]
    if not ordered_uids:
        empty = _empty_corr_figure("No selected strategies with P&L data.")
        return empty, empty

    contr_vals = contr_vals.reindex(ordered_uids)
    labels_full = [uid_to_name[u] for u in ordered_uids]
    labels_axis = [
        (name[:25] + "…") if len(name) > 25 else name
        for name in labels_full
    ]
    
    # Waterfall: one bar per strategy, final total
    bar_vals = list(contr_vals.values)
    total_val = float(contr_vals.sum())
    
    wf_x = labels_axis + ["Total"]
    wf_y = bar_vals + [total_val]
    measures = ["relative"] * len(bar_vals) + ["total"]
    
    # Build contribution + cumulative explicitly for hover
    contrib_vals: list[float] = []
    cumulative_vals: list[float] = []
    running = 0.0
    for i, val in enumerate(wf_y):
        if measures[i] == "total":
            contrib = val          # total bar
            running = val
        else:
            contrib = val          # relative step = contribution
            running += val
        contrib_vals.append(contrib)
        cumulative_vals.append(running)
    
    hover_names = labels_full + ["Total"]
    
    # customdata: col 0 = contribution, col 1 = cumulative
    customdata = np.column_stack([contrib_vals, cumulative_vals])
    
    if pnl_mode == "pct":
        hovertemplate = (
            "<b>%{hovertext}</b><br>"
            "Contribution: %{customdata[0]:.2f}%<br>"
            "Cumulative: %{customdata[1]:.2f}%"
            "<extra></extra>"
        )
    else:
        hovertemplate = (
            "<b>%{hovertext}</b><br>"
            "Contribution: %{customdata[0]:,.0f}<br>"
            "Cumulative: %{customdata[1]:,.0f}"
            "<extra></extra>"
        )
    
    pnl_fig = go.Figure()
    pnl_fig.add_trace(
        go.Waterfall(
            x=wf_x,
            measure=measures,
            y=wf_y,
            increasing={"marker": {"color": "#2ECC71"}},
            decreasing={"marker": {"color": "#E74C3C"}},
            totals={"marker": {"color": "#5DADE2"}},
            connector={"line": {"color": "#888888"}},
            hovertext=hover_names,
            customdata=customdata,
            hovertemplate=hovertemplate,
        )
    )


    pnl_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        showlegend=False,
        margin=dict(l=40, r=10, t=40, b=60),
        yaxis_title=y_title_pnl,
    )

    # ---------- DD contribution (DD-worsening days only) ----------
    dd_contrib = _compute_dd_contribution_from_series(series)
    dd_contrib = dd_contrib.reindex(ordered_uids)

    # Sort worst (most negative) to best
    dd_contrib_sorted = dd_contrib.sort_values()

    bar_colors = [
        "#E74C3C" if v < 0 else "#2ECC71" for v in dd_contrib_sorted.values
    ]

    dd_fig = go.Figure()
    dd_fig.add_trace(
        go.Bar(
            x=dd_contrib_sorted.values,
            y=[uid_to_name[u] for u in dd_contrib_sorted.index],
            orientation="h",
            marker={"color": bar_colors},
        )
    )
    dd_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        showlegend=False,
        margin=dict(l=80, r=20, t=40, b=40),
        xaxis_title="DD contribution on DD-worsening days ($)",
    )

    return pnl_fig, dd_fig


# ---------------------------------------------------------------------------
# Phase 2 – Correlation helpers (CORR1–CORR4)
# ---------------------------------------------------------------------------

def _build_corr_matrices(
    active_store: list,
    weights_store: dict | None,
    initial_equity: float,
) -> dict | None:
    """
    Build correlation-related matrices using daily P&L per strategy.

    Uses _build_portfolio_timeseries to:
      - get aligned daily P&L matrix (pnl_df) for selected strategies
      - keep strategy IDs consistent with Phase 2.

    Returns dict with:
      - uids: list of strategy UIDs in matrix order
      - labels_full: list of full display names (same order as uids)
      - labels_short: list of truncated names (max 15 chars) for axes
      - codes: list of short codes 'S1', 'S2', ... aligned with uids
      - pearson: DataFrame (CORR1)
      - downside: DataFrame (CORR2)
      - tail: DataFrame (CORR3)
      - kendall_avg: Series indexed by UID (CORR4)
      - spearman_avg: Series indexed by UID (CORR4)
    """
    # Reuse portfolio engine to build aligned P&L
    series = _build_portfolio_timeseries(
        active_store=active_store,
        weights_store=weights_store,
        weight_mode="factors",
        initial_equity=float(initial_equity or 100000.0),
    )
    if series is None:
        return None

    pnl_df: pd.DataFrame = series["pnl_df"].copy()
    if pnl_df.empty or pnl_df.shape[1] < 1:
        return None

    # Map uid -> display name based on Active list
    uid_to_name: dict[str, str] = {}
    active_store = active_store or []
    for row in active_store:
        if not row.get("is_selected", False):
            continue
        uid = row.get("uid")
        if not uid:
            continue
        raw_name = row.get("name") or uid or row.get("sid")
        uid_to_name[uid] = raw_name

    # Keep only selected UIDs present in pnl_df, preserve order
    ordered_uids = [
        uid for uid in pnl_df.columns if uid in uid_to_name
    ]
    if len(ordered_uids) < 1:
        return None
    pnl_df = pnl_df[ordered_uids]

    labels_full = [uid_to_name[uid] for uid in ordered_uids]

    labels_short: list[str] = []
    for name in labels_full:
        if len(name) > 15:
            labels_short.append(name[:15] + "…")
        else:
            labels_short.append(name)

    # Codes S1, S2, ..., aligned with uids
    codes = [f"S{i+1}" for i in range(len(ordered_uids))]

    # ---------------- CORR1 – Pearson correlation -------------------
    pearson = pnl_df.corr(method="pearson")
    if pearson.size > 0:
        np.fill_diagonal(pearson.values, 1.0)

    # ---------------- CORR2 – Downside correlation -----------------
    downside = pd.DataFrame(
        np.nan, index=pearson.index, columns=pearson.columns
    )

    # ---------------- CORR3 – Tail co-crash frequency --------------
    tail = pd.DataFrame(
        np.nan, index=pearson.index, columns=pearson.columns
    )

    cols = list(pnl_df.columns)
    for i, c1 in enumerate(cols):
        x = pnl_df[c1]
        for j, c2 in enumerate(cols[i:], start=i):
            y = pnl_df[c2]

            # Common non-NaN days
            mask_common = x.notna() & y.notna()
            n_common = int(mask_common.sum())
            if n_common < 5:
                continue

            x_common = x[mask_common]
            y_common = y[mask_common]

            # Downside correlation: both < 0
            mask_down = (x_common < 0) & (y_common < 0)
            n_down = int(mask_down.sum())
            if n_down >= 3:
                x_down = x_common[mask_down].astype(float)
                y_down = y_common[mask_down].astype(float)

                # Need at least 2 points and non-zero variance on both sides
                if x_down.shape[0] >= 2:
                    sx = float(x_down.std(ddof=0))
                    sy = float(y_down.std(ddof=0))
                    if sx > 0 and sy > 0:
                        corr_down = float(
                            np.corrcoef(x_down.values, y_down.values)[0, 1]
                        )
                    else:
                        corr_down = np.nan
                else:
                    corr_down = np.nan

                downside.loc[c1, c2] = corr_down
                downside.loc[c2, c1] = corr_down


            # Tail co-crash: both below own 5% quantile
            try:
                qx = float(x_common.quantile(0.05))
                qy = float(y_common.quantile(0.05))
            except Exception:
                continue

            mask_tail = (x_common < qx) & (y_common < qy)
            n_tail = int(mask_tail.sum())
            if n_common > 0:
                p_tail = n_tail / n_common
                tail.loc[c1, c2] = p_tail
                tail.loc[c2, c1] = p_tail

    # Fill diagonals with 1.0 where missing
    if downside.size > 0:
        for k in range(len(downside)):
            downside.iat[k, k] = 1.0
    if tail.size > 0:
        for k in range(len(tail)):
            tail.iat[k, k] = 1.0

    # ---------------- CORR4 – Rank correlations (avg) --------------
    kendall = pnl_df.corr(method="kendall")
    spearman = pnl_df.corr(method="spearman")

    def _avg_off_diag(mat: pd.DataFrame) -> pd.Series:
        if mat is None or mat.empty:
            return pd.Series(dtype=float)
        out = {}
        for col in mat.columns:
            s = mat[col].drop(labels=[col], errors="ignore").dropna()
            out[col] = float(s.mean()) if not s.empty else np.nan
        return pd.Series(out)

    kendall_avg = _avg_off_diag(kendall)     # index = UID
    spearman_avg = _avg_off_diag(spearman)   # index = UID

    return {
        "uids": ordered_uids,
        "labels_full": labels_full,
        "labels_short": labels_short,
        "codes": codes,
        "pearson": pearson,
        "downside": downside,
        "tail": tail,
        "kendall_avg": kendall_avg,
        "spearman_avg": spearman_avg,
    }




def _empty_corr_figure(message: str) -> go.Figure:
    """
    Build an empty dark-themed figure with a centered annotation message.
    Used when <2 strategies are selected or no data.
    """
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 12, "color": "#AAAAAA"},
            )
        ],
        margin=dict(l=40, r=10, t=30, b=30),
    )
    return fig


def _corr_heatmap_figure(
    mat: pd.DataFrame,
    uids: list[str],
    labels_axis: list[str],
    labels_full: list[str],
    title: str,
    zmin: float | None = None,
    zmax: float | None = None,
    zmid: float | None = None,
    zfmt: str = ".2f",
) -> go.Figure:
    """
    Generic helper to render a correlation-style heatmap in dark theme.

    - Axis tick labels use labels_axis (either truncated names or codes S1..Sn).
    - Hover shows full strategy names via customdata.
    """
    if (
        mat is None
        or mat.empty
        or uids is None
        or len(uids) < 2
        or len(labels_axis) != len(uids)
        or len(labels_full) != len(uids)
    ):
        return _empty_corr_figure("Select at least two strategies to see correlation.")

    # Ensure matrix order follows UID list
    mat = mat.reindex(index=uids, columns=uids)
    z = mat.values

    n = len(uids)
    # Build customdata with full names for (i, j)
    customdata = np.empty((n, n, 2), dtype=object)
    for i in range(n):
        for j in range(n):
            customdata[i, j, 0] = labels_full[i]  # row / y
            customdata[i, j, 1] = labels_full[j]  # col / x

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=labels_axis,
            y=labels_axis,
            customdata=customdata,
            colorscale="RdBu_r",
            zmin=zmin,
            zmax=zmax,
            zmid=zmid,
            colorbar=dict(title=""),
            hovertemplate=(
                "Strategy i: %{customdata[0]}<br>"
                "Strategy j: %{customdata[1]}<br>"
                f"Value: %{{z:{zfmt}}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin=dict(l=60, r=20, t=40, b=60),
        xaxis=dict(ticks="", showgrid=False),
        yaxis=dict(ticks="", showgrid=False, autorange="reversed"),
    )
    return fig




def _corr_alt_table_component(
    uids: list[str],
    labels_full: list[str],
    codes: list[str],
    kendall_avg: pd.Series,
    spearman_avg: pd.Series,
) -> html.Div:
    """
    Build the CORR4 HTML table showing avg Kendall / Spearman per strategy.

    Uses:
      - uids: list of strategy UIDs in matrix order
      - labels_full: corresponding full names
      - codes: corresponding short codes (S1..Sn)
      - kendall_avg / spearman_avg: Series indexed by UID
    """
    if (
        not uids
        or not labels_full
        or not codes
        or kendall_avg is None
        or spearman_avg is None
    ):
        return html.Div(
            "Select at least two strategies to see rank correlation summary.",
            style={"fontSize": "0.8rem", "color": "#AAAAAA"},
        )

    rows = []
    header = html.Tr(
        [
            html.Th("Code", style={"padding": "0.2rem 0.4rem"}),
            html.Th("Strategy", style={"padding": "0.2rem 0.4rem"}),
            html.Th("Avg Kendall τ", style={"padding": "0.2rem 0.4rem"}),
            html.Th("Avg Spearman ρ", style={"padding": "0.2rem 0.4rem"}),
        ]
    )

    for uid, full_name, code in zip(uids, labels_full, codes):
        k_val = float(kendall_avg.get(uid, np.nan))
        s_val = float(spearman_avg.get(uid, np.nan))

        row = html.Tr(
            [
                html.Td(code, style={"padding": "0.2rem 0.4rem"}),
                html.Td(full_name, style={"padding": "0.2rem 0.4rem"}),
                html.Td(
                    "N/A" if np.isnan(k_val) else f"{k_val:.3f}",
                    style={"padding": "0.2rem 0.4rem"},
                ),
                html.Td(
                    "N/A" if np.isnan(s_val) else f"{s_val:.3f}",
                    style={"padding": "0.2rem 0.4rem"},
                ),
            ]
        )
        rows.append(row)

    table = html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={
            "width": "100%",
            "borderCollapse": "collapse",
        },
    )
    return html.Div(table)


# ---------------------------------------------------------------------------
# Phase 2 – CORR5 helper: strategy vs portfolio by VIX regime
# ---------------------------------------------------------------------------
def _build_corr5_vix_matrix(
    active_store: list,
    weights_store: dict | None,
    initial_equity: float,
    vix_mode: str = "auto",
    manual_bounds: list[float] | None = None,
) -> dict | None:
    """
    Build CORR5 matrix: for each strategy and each VIX regime bucket, compute
    the Pearson correlation between strategy daily P&L and the portfolio daily
    P&L restricted to that regime.

    Regimes:
      - AUTO  : quartiles of daily entry-based Opening VIX.
      - MANUAL: user-specified three cut levels (low → high) from sliders.

    Returns dict with:
      - uids         : list of strategy UIDs
      - labels_full  : full strategy names (same order as uids)
      - labels_axis  : y-axis labels (truncated or S1..Sn)
      - bucket_labels: list of x-axis labels for VIX regimes
      - corr         : DataFrame shape (n_strats, 4) with correlations
    """
    # Build daily P&L matrix and portfolio series
    series = _build_portfolio_timeseries(
        active_store=active_store or [],
        weights_store=weights_store or {},
        weight_mode="factors",
        initial_equity=float(initial_equity or 100000.0),
    )
    if series is None:
        return None

    pnl_df: pd.DataFrame = series["pnl_df"].copy()
    portfolio_daily: pd.Series = series["portfolio_daily"]
    dates: pd.DatetimeIndex = series["dates"]

    if pnl_df.empty or pnl_df.shape[1] < 1:
        return None

    # Map uid -> display name based on Active list (selected only)
    uid_to_name: dict[str, str] = {}
    active_store = active_store or []
    for row in active_store:
        if not row.get("is_selected", False):
            continue
        uid = row.get("uid")
        if not uid:
            continue
        raw_name = row.get("name") or uid or row.get("sid")
        uid_to_name[uid] = raw_name

    # Keep only selected UIDs present in pnl_df, preserve order
    ordered_uids = [uid for uid in pnl_df.columns if uid in uid_to_name]
    if len(ordered_uids) < 1:
        return None

    pnl_df = pnl_df[ordered_uids]
    labels_full = [uid_to_name[uid] for uid in ordered_uids]

    # Short names (for <=10 strategies)
    labels_short: list[str] = []
    for name in labels_full:
        if len(name) > 15:
            labels_short.append(name[:15] + "…")
        else:
            labels_short.append(name)

    codes = [f"S{i+1}" for i in range(len(ordered_uids))]

    # Decide y-axis labels consistent with CORR1–CORR4
    if len(ordered_uids) <= 10:
        labels_axis = labels_short
    else:
        labels_axis = codes

    # ------------------------------------------------------------------
    # Build daily Opening VIX per date (entry-based) from strategy CSVs
    # ------------------------------------------------------------------
    date_to_vix: dict[object, list[float]] = {}
    for row in active_store:
        if not row.get("is_selected", False):
            continue
        uid = row.get("uid")
        if not uid:
            continue
        meta = p1_strategy_store.get(uid, {})
        df = meta.get("df")
        if df is None or df.empty:
            continue
        if "Date Closed" not in df.columns or "Opening VIX" not in df.columns:
            continue

        tmp = df[["Date Closed", "Opening VIX"]].copy()
        tmp["Date Closed"] = pd.to_datetime(tmp["Date Closed"]).dt.date
        tmp = tmp.dropna(subset=["Opening VIX"])
        if tmp.empty:
            continue

        for d, v in zip(tmp["Date Closed"], tmp["Opening VIX"]):
            try:
                v_float = float(v)
            except (TypeError, ValueError):
                continue
            if np.isnan(v_float):
                continue
            date_to_vix.setdefault(d, []).append(v_float)

    if not date_to_vix:
        return None

    # Align VIX to portfolio date index
    vix_values: list[float] = []
    for ts in dates:
        d = ts.date()
        vals = date_to_vix.get(d)
        if vals:
            vix_values.append(float(np.mean(vals)))
        else:
            vix_values.append(np.nan)

    vix_series = pd.Series(vix_values, index=dates, name="Opening VIX").dropna()
    if vix_series.empty:
        return None

    # ------------------------------------------------------------------
    # Define VIX regimes: AUTO (quartiles) or MANUAL (slider bounds)
    # ------------------------------------------------------------------
    def _bucket(v: float) -> int:
        if v <= b1:
            return 0
        elif v <= b2:
            return 1
        elif v <= b3:
            return 2
        else:
            return 3
        
    vix_mode = (vix_mode or "auto").lower()
    
    use_manual = False
    b1 = b2 = b3 = None
    if vix_mode == "manual" and manual_bounds and len(manual_bounds) == 3:
        try:
            bounds_sorted = sorted(float(x) for x in manual_bounds)
            b1, b2, b3 = bounds_sorted
            use_manual = True
        except Exception:
            use_manual = False

    if use_manual:
        # MANUAL: user-defined boundaries
        bucket_labels = [
            f"R1\nVIX ≤ {b1:.1f}",
            f"R2\n{b1:.1f}–{b2:.1f}",
            f"R3\n{b2:.1f}–{b3:.1f}",
            f"R4\nVIX ≥ {b3:.1f}",
        ]
    else:
        # AUTO: quartiles
        q = vix_series.quantile([0.25, 0.5, 0.75])
        q1, q2, q3 = float(q.iloc[0]), float(q.iloc[1]), float(q.iloc[2])

        def _bucket(v: float) -> int:
            if v <= q1:
                return 0
            elif v <= q2:
                return 1
            elif v <= q3:
                return 2
            else:
                return 3

        bucket_labels = [
            f"Q1\nVIX ≤ {q1:.1f}",
            f"Q2\n{q1:.1f}–{q2:.1f}",
            f"Q3\n{q2:.1f}–{q3:.1f}",
            f"Q4\nVIX ≥ {q3:.1f}",
        ]

    bucket_series = vix_series.apply(_bucket)
    buckets = [0, 1, 2, 3]

    data = np.full((len(ordered_uids), len(buckets)), np.nan, dtype=float)

    # Compute correlation within each regime bucket
    for b_idx, b in enumerate(buckets):
        mask = bucket_series == b
        if int(mask.sum()) < 5:
            continue

        idx_b = mask.index[mask]
        # Full series for this bucket (with possible NaNs)
        port_b_full = portfolio_daily.reindex(idx_b).astype(float)

        for i, uid in enumerate(ordered_uids):
            # Strategy series for this bucket
            s_full = pnl_df[uid].reindex(idx_b).astype(float)

            # Drop any rows where either side is NaN
            pair = pd.concat([s_full, port_b_full], axis=1).dropna()
            if pair.shape[0] < 5:
                corr_val = np.nan
            else:
                x = pair.iloc[:, 0].values  # strategy
                y = pair.iloc[:, 1].values  # portfolio

                var_x = float(np.var(x))
                var_y = float(np.var(y))
                if var_x == 0.0 or var_y == 0.0:
                    corr_val = np.nan
                else:
                    cov_xy = float(np.cov(x, y, ddof=0)[0, 1])
                    # Pearson correlation = cov / sqrt(var_x * var_y)
                    corr_val = cov_xy / np.sqrt(var_x * var_y)

            data[i, b_idx] = corr_val


    corr = pd.DataFrame(
        data,
        index=ordered_uids,
        columns=buckets,
    )

    return {
        "uids": ordered_uids,
        "labels_full": labels_full,
        "labels_axis": labels_axis,
        "bucket_labels": bucket_labels,
        "corr": corr,
    }



def _corr5_vix_heatmap_figure(info: dict | None) -> go.Figure:
    """
    Render CORR5 matrix as a heatmap.

    Y-axis: strategies (truncated names or S1..Sn).
    X-axis: VIX regime buckets (quartiles of Opening VIX).
    """
    if not info:
        return _empty_corr_figure(
            "Select at least one strategy with Opening VIX data to see CORR5."
        )

    corr: pd.DataFrame = info["corr"]
    uids: list[str] = info["uids"]
    labels_axis: list[str] = info["labels_axis"]
    labels_full: list[str] = info["labels_full"]
    bucket_labels: list[str] = info["bucket_labels"]

    if corr is None or corr.empty or len(uids) == 0:
        return _empty_corr_figure(
            "Select at least one strategy with Opening VIX data to see CORR5."
        )

    # Ensure row order consistent with uids
    corr = corr.reindex(index=uids)
    z = corr.values
    n_strats = len(uids)
    n_buckets = len(bucket_labels)

    # customdata: full strategy name + regime label
    customdata = np.empty((n_strats, n_buckets, 2), dtype=object)
    for i in range(n_strats):
        for j in range(n_buckets):
            customdata[i, j, 0] = labels_full[i]
            customdata[i, j, 1] = bucket_labels[j]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=bucket_labels,
            y=labels_axis,
            customdata=customdata,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            zmid=0,
            colorbar=dict(title="ρ"),
            hovertemplate=(
                "Strategy: %{customdata[0]}<br>"
                "Regime: %{customdata[1]}<br>"
                "Corr: %{z:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="CORR5 – Strategy vs Portfolio correlation by VIX regime",
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin=dict(l=60, r=20, t=40, b=60),
        xaxis=dict(title="Opening VIX regime (quartiles)"),
        yaxis=dict(title="Strategy", autorange="reversed"),
    )
    return fig

# ---------------------------------------------------------------------------
# CORR6 – Drawdown overlap matrix helper
# ---------------------------------------------------------------------------

def _build_corr6_ddoverlap_matrix(
    pnl_df: pd.DataFrame,
    uids: list[str],
) -> pd.DataFrame | None:
    """
    CORR6 helper – build drawdown–overlap matrix.

    Parameters
    ----------
    pnl_df : DataFrame
        Daily P&L per strategy, columns = strategy uids, index = dates.
    uids : list[str]
        Full ordered list of strategy uids we want in the matrix.

    Returns
    -------
    DataFrame or None
        Square matrix with index/columns = uids. Values in [0,1] are
        overlap fractions; 0 on diagonal if a strategy never has a
        drawdown.
    """
    if pnl_df is None or pnl_df.empty:
        return None

    # Keep only columns we actually have, in the requested order
    cols = [uid for uid in uids if uid in pnl_df.columns]
    if len(cols) < 2:
        return None

    pnl = pnl_df[cols].copy()

    # Equity, running max, drawdown, "in drawdown" mask
    equity = pnl.cumsum()
    running_max = equity.cummax()
    dd = running_max - equity
    in_dd = dd > 0

    n = len(cols)
    overlap = np.zeros((n, n), dtype=float)

    for i in range(n):
        mi = in_dd.iloc[:, i]
        for j in range(n):
            mj = in_dd.iloc[:, j]
            union = mi | mj
            if union.sum() == 0:
                overlap[i, j] = 0.0
            else:
                inter = mi & mj
                overlap[i, j] = float(inter.sum()) / float(union.sum())

    # Diagonal: if a strategy never has a drawdown, set to 0 instead of 1
    for k in range(n):
        if in_dd.iloc[:, k].any():
            overlap[k, k] = 1.0
        else:
            overlap[k, k] = 0.0

    mat = pd.DataFrame(overlap, index=cols, columns=cols)

    # Expand to full uids list so _corr_heatmap_figure can reorder safely
    mat = mat.reindex(index=uids, columns=uids)

    return mat


# ---------------------------------------------------------------------------
# CORR7 – Drawdown depth correlation helper
# ---------------------------------------------------------------------------

def _build_corr7_dddepth_matrix(pnl_df: pd.DataFrame, uids: list[str]) -> pd.DataFrame | None:
    """
    Build CORR7 matrix: correlation of drawdown depths between strategies.

    For each strategy i:
      equity_i = pnl_i.cumsum()
      max_i    = equity_i.cummax()
      dd_i     = max_i - equity_i
      mask_i   = dd_i > 0  (days in drawdown)

    For each pair (i, j), we compute Pearson corr(dd_i, dd_j) restricted to
    days where BOTH are in drawdown (mask_i & mask_j).
    """
    if pnl_df is None or pnl_df.empty:
        return None

    cols = [uid for uid in uids if uid in pnl_df.columns]
    if len(cols) < 2:
        return None

    pnl = pnl_df[cols].copy()

    equity = pnl.cumsum()
    running_max = equity.cummax()
    dd = running_max - equity  # same shape as pnl

    n = len(cols)
    mat = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        uid_i = cols[i]
        dd_i = dd[uid_i]
        mask_i = dd_i > 0

        for j in range(i, n):
            uid_j = cols[j]
            dd_j = dd[uid_j]
            mask_j = dd_j > 0

            mask_both = mask_i & mask_j
            n_both = int(mask_both.sum())
            if n_both < 5:
                val = np.nan
            else:
                x_full = dd_i[mask_both].astype(float)
                y_full = dd_j[mask_both].astype(float)

                # Drop any rows with NaNs in either series
                pair = pd.concat([x_full, y_full], axis=1).dropna()
                if pair.shape[0] < 2:
                    val = np.nan
                else:
                    x = pair.iloc[:, 0].values
                    y = pair.iloc[:, 1].values

                    var_x = float(np.var(x))
                    var_y = float(np.var(y))
                    if var_x == 0.0 or var_y == 0.0:
                        val = np.nan
                    else:
                        cov_xy = float(np.cov(x, y, ddof=0)[0, 1])
                        val = cov_xy / np.sqrt(var_x * var_y)

            mat[i, j] = val
            mat[j, i] = val


    mat_df = pd.DataFrame(mat, index=cols, columns=cols)
    mat_df = mat_df.reindex(index=uids, columns=uids)
    return mat_df


# ---------------------------------------------------------------------------
# BETA1 – Beta vs portfolio helpers
# ---------------------------------------------------------------------------

def _build_beta_vs_portfolio(
    active_store: list,
    weights_store: dict | None,
    initial_equity: float,
) -> tuple[pd.DataFrame | None, list[str] | None, list[str] | None, list[str] | None]:
    """
    Compute beta of each strategy's daily P&L versus the portfolio daily P&L.

    Returns:
      beta_df: DataFrame with columns ['uid', 'name', 'beta', 'r2']
      uids: ordered list of strategy uids used
      labels_full: names for each uid
      axis_labels: labels for charts (short names or S1..Sn)
    """
    series = _build_portfolio_timeseries(
        active_store=active_store or [],
        weights_store=weights_store or {},
        weight_mode="factors",
        initial_equity=float(initial_equity or 100000.0),
    )
    if series is None:
        return None, None, None, None

    pnl_df: pd.DataFrame = series["pnl_df"].copy()
    port: pd.Series = series["portfolio_daily"]
    if pnl_df.empty:
        return None, None, None, None

    # Map uid -> name for selected strategies
    uid_to_name: dict[str, str] = {}
    active_store = active_store or []
    for row in active_store:
        if not row.get("is_selected", False):
            continue
        uid = row.get("uid")
        if not uid:
            continue
        raw_name = row.get("name") or uid or row.get("sid")
        uid_to_name[uid] = raw_name

    uids = [uid for uid in pnl_df.columns if uid in uid_to_name]
    if len(uids) < 1:
        return None, None, None, None

    labels_full = [uid_to_name[uid] for uid in uids]
    labels_short: list[str] = []
    for name in labels_full:
        labels_short.append(name[:15] + "…" if len(name) > 15 else name)

    codes = [f"S{i+1}" for i in range(len(uids))]
    axis_labels = labels_short if len(uids) <= 10 else codes

    betas = []
    for uid in uids:
        s = pnl_df[uid].astype(float)
        df = pd.concat([s, port], axis=1, join="inner").dropna()
        if df.shape[0] < 20:
            betas.append((uid, uid_to_name[uid], np.nan, np.nan))
            continue

        x = df.iloc[:, 1].values  # portfolio
        y = df.iloc[:, 0].values  # strategy

        var_x = np.var(x)
        var_y = np.var(y)

        # If either side has zero variance, beta / R² are not meaningful
        if var_x == 0 or var_y == 0:
            betas.append((uid, uid_to_name[uid], np.nan, np.nan))
            continue

        cov_xy = np.cov(x, y, ddof=0)[0, 1]
        beta = cov_xy / var_x

        # R² via covariance / variance, without np.corrcoef
        # corr^2 = cov_xy^2 / (var_x * var_y)
        r2 = float((cov_xy ** 2) / (var_x * var_y))

        betas.append((uid, uid_to_name[uid], float(beta), r2))


    beta_df = pd.DataFrame(
        betas, columns=["uid", "name", "beta", "r2"]
    )
    return beta_df, uids, labels_full, axis_labels


def _beta_summary_table_component(beta_df: pd.DataFrame) -> html.Div:
    """
    Render a small HTML table with beta and R² values.
    """
    if beta_df is None or beta_df.empty:
        return html.Div("No data to compute betas.", style={"color": "#AAAAAA"})

    header = html.Thead(
        html.Tr(
            [
                html.Th("Strategy", style={"textAlign": "left", "padding": "2px 4px"}),
                html.Th("β vs PF", style={"textAlign": "right", "padding": "2px 4px"}),
                html.Th("R²", style={"textAlign": "right", "padding": "2px 4px"}),
            ]
        )
    )

    body_rows = []
    for _, row in beta_df.iterrows():
        body_rows.append(
            html.Tr(
                [
                    html.Td(row["name"], style={"padding": "2px 4px"}),
                    html.Td(
                        f"{row['beta']:.2f}" if pd.notnull(row["beta"]) else "—",
                        style={"textAlign": "right", "padding": "2px 4px"},
                    ),
                    html.Td(
                        f"{row['r2']:.2f}" if pd.notnull(row["r2"]) else "—",
                        style={"textAlign": "right", "padding": "2px 4px"},
                    ),
                ]
            )
        )

    table = html.Table(
        [header, html.Tbody(body_rows)],
        style={
            "width": "100%",
            "borderCollapse": "collapse",
        },
    )
    return html.Div(table)


# ---------------------------------------------------------------------------
# SD1 – Serial dependence helpers (lag-1 autocorrelation)
# ---------------------------------------------------------------------------

def _build_serial_ac(
    active_store: list,
    weights_store: dict | None,
    initial_equity: float,
) -> tuple[pd.Series | None, list[str] | None, list[str] | None, list[str] | None]:
    """
    Compute lag-1 autocorrelation of daily P&L for each selected strategy.

    Returns:
      ac_series: Series indexed by uid with lag-1 autocorr
      uids: ordered list of uids
      labels_full: full names
      axis_labels: labels for bar chart (short names or S1..Sn)
    """
    series = _build_portfolio_timeseries(
        active_store=active_store or [],
        weights_store=weights_store or {},
        weight_mode="factors",
        initial_equity=float(initial_equity or 100000.0),
    )
    if series is None:
        return None, None, None, None

    pnl_df: pd.DataFrame = series["pnl_df"].copy()
    if pnl_df.empty:
        return None, None, None, None

    uid_to_name: dict[str, str] = {}
    active_store = active_store or []
    for row in active_store:
        if not row.get("is_selected", False):
            continue
        uid = row.get("uid")
        if not uid:
            continue
        raw_name = row.get("name") or uid or row.get("sid")
        uid_to_name[uid] = raw_name

    uids = [uid for uid in pnl_df.columns if uid in uid_to_name]
    if len(uids) < 1:
        return None, None, None, None

    labels_full = [uid_to_name[uid] for uid in uids]
    labels_short: list[str] = []
    for name in labels_full:
        labels_short.append(name[:15] + "…" if len(name) > 15 else name)

    codes = [f"S{i+1}" for i in range(len(uids))]
    axis_labels = labels_short if len(uids) <= 10 else codes

    ac_values = {}
    for uid in uids:
        s = pnl_df[uid].astype(float).dropna()
        if s.shape[0] < 20:
            ac_values[uid] = np.nan
            continue
        try:
            ac = float(s.autocorr(lag=1))
        except Exception:
            ac = np.nan
        ac_values[uid] = ac

    ac_series = pd.Series(ac_values)
    return ac_series, uids, labels_full, axis_labels


# ---------------------------------------------------------------------------
# Phase 2 – Correlation callback (CORR1–CORR4)
# ---------------------------------------------------------------------------

@callback(
    Output("p2-corr-pearson-heatmap", "figure"),
    Output("p2-corr-downside-heatmap", "figure"),
    Output("p2-corr-taildep-heatmap", "figure"),
    Output("p2-corr-alt-table", "children"),
    Input("p1-active-list-store", "data"),
    Input("p2-weights-store", "data"),
    Input("p2-initial-equity-input", "value"),
)
def update_portfolio_correlation(
    active_store,
    weights_store,
    initial_equity,
):
    """
    Phase 2 – Correlation tab engine.

    Builds:
      - CORR1: Pearson correlation heatmap
      - CORR2: Downside correlation heatmap
      - CORR3: Tail co-crash frequency heatmap
      - CORR4: Rank correlation summary table (Kendall / Spearman)

    Axis labels:
      - If <=10 strategies: truncated names (labels_short).
      - If >10 strategies: codes S1, S2, ..., Sn.
    """
    active_store = active_store or []
    weights_store = weights_store or {}

    # If <2 selected strategies, short-circuit with empty figs
    selected_rows = [r for r in active_store if r.get("is_selected")]
    if len(selected_rows) < 2:
        msg = "Select at least two strategies in the Active list to see correlation."
        empty_fig = _empty_corr_figure(msg)
        alt_table = html.Div(
            msg,
            style={"fontSize": "0.8rem", "color": "#AAAAAA"},
        )
        return empty_fig, empty_fig, empty_fig, alt_table

    corr_data = _build_corr_matrices(
        active_store=active_store,
        weights_store=weights_store,
        initial_equity=initial_equity,
    )
    if corr_data is None:
        msg = "No overlapping daily P&L data for selected strategies."
        empty_fig = _empty_corr_figure(msg)
        alt_table = html.Div(
            msg,
            style={"fontSize": "0.8rem", "color": "#AAAAAA"},
        )
        return empty_fig, empty_fig, empty_fig, alt_table

    uids = corr_data["uids"]
    labels_full = corr_data["labels_full"]
    labels_short = corr_data["labels_short"]
    codes = corr_data["codes"]
    pearson = corr_data["pearson"]
    downside = corr_data["downside"]
    tail = corr_data["tail"]
    kendall_avg = corr_data["kendall_avg"]
    spearman_avg = corr_data["spearman_avg"]

    n_strat = len(uids)
    # Dynamic axis labels: names if <=10, codes if >10
    if n_strat <= 10:
        axis_labels = labels_short
    else:
        axis_labels = codes

    # ---------------- Build figures --------------------------
    pearson_fig = _corr_heatmap_figure(
        mat=pearson,
        uids=uids,
        labels_axis=axis_labels,
        labels_full=labels_full,
        title="CORR1 – Pearson correlation (daily P&L)",
        zmin=-1.0,
        zmax=1.0,
        zmid=0.0,
        zfmt=".2f",
    )

    downside_fig = _corr_heatmap_figure(
        mat=downside,
        uids=uids,
        labels_axis=axis_labels,
        labels_full=labels_full,
        title="CORR2 – Downside correlation (negative days only)",
        zmin=-1.0,
        zmax=1.0,
        zmid=0.0,
        zfmt=".2f",
    )

    tail_fig = _corr_heatmap_figure(
        mat=tail,
        uids=uids,
        labels_axis=axis_labels,
        labels_full=labels_full,
        title="CORR3 – Tail co-crash frequency (worst 5% days)",
        zmin=0.0,
        zmax=1.0,
        zmid=0.5,
        zfmt=".2f",
    )

    alt_table = _corr_alt_table_component(
        uids=uids,
        labels_full=labels_full,
        codes=codes,
        kendall_avg=kendall_avg,
        spearman_avg=spearman_avg,
    )

    return pearson_fig, downside_fig, tail_fig, alt_table


# ---------------------------------------------------------------------------
# Phase 2 – CORR5 callback (strategy vs portfolio by VIX regime)
# ---------------------------------------------------------------------------
@callback(
    Output("p2-corr5-vix-heatmap", "figure"),
    Input("p1-active-list-store", "data"),
    Input("p2-weights-store", "data"),
    Input("p2-initial-equity-input", "value"),
    Input("p2-corr5-vix-mode-auto", "n_clicks"),
    Input("p2-corr5-vix-mode-manual", "n_clicks"),
    Input("p2-corr5-vix-slider-1", "value"),
    Input("p2-corr5-vix-slider-2", "value"),
    Input("p2-corr5-vix-slider-3", "value"),
)
def update_corr5_vix_regime(
    active_store,
    weights_store,
    initial_equity,
    n_auto,
    n_manual,
    v1,
    v2,
    v3,
):
    """
    Update CORR5 heatmap: correlation of each strategy vs portfolio by Opening
    VIX regime.

    - AUTO   : regimes based on VIX quartiles.
    - MANUAL : regimes based on slider boundaries v1 < v2 < v3.
    """
    active_store = active_store or []
    weights_store = weights_store or {}
    initial_equity = float(initial_equity or 100000.0)

    # Determine mode consistent with toggle_corr5_vix_mode
    n_auto = n_auto or 0
    n_manual = n_manual or 0
    manual_mode = n_manual > n_auto

    vix_mode = "auto"
    bounds: list[float] | None = None

    if manual_mode:
        # Use manual bounds only if all three sliders have values
        raw_bounds = [v1, v2, v3]
        if all(b is not None for b in raw_bounds):
            try:
                bounds = sorted(float(b) for b in raw_bounds)
                vix_mode = "manual"
            except Exception:
                # Fallback to AUTO if parsing fails
                vix_mode = "auto"
                bounds = None

    info = _build_corr5_vix_matrix(
        active_store=active_store,
        weights_store=weights_store,
        initial_equity=initial_equity,
        vix_mode=vix_mode,
        manual_bounds=bounds,
    )

    return _corr5_vix_heatmap_figure(info)


# ---------------------------------------------------------------------------
# Phase 2 – CORR6 callback (drawdown overlap matrix)
# ---------------------------------------------------------------------------

@callback(
    Output("p2-corr-ddoverlap-heatmap", "figure"),
    Input("p1-active-list-store", "data"),
    Input("p2-weights-store", "data"),
    Input("p2-initial-equity-input", "value"),
)
def update_corr6_ddoverlap(
    active_store,
    weights_store,
    initial_equity,
):
    """
    Update CORR6 heatmap: drawdown-overlap matrix between strategies.

    Uses each strategy's own 1× equity curve (daily P&L cumsum); weights do not
    affect which days are in drawdown, so sizing is ignored here.
    """
    active_store = active_store or []
    weights_store = weights_store or {}
    initial_equity = float(initial_equity or 100000.0)

    # Need at least 2 selected strategies
    selected_rows = [r for r in active_store if r.get("is_selected")]
    if len(selected_rows) < 2:
        msg = "Select at least two strategies in the Active list to see drawdown overlap."
        return _empty_corr_figure(msg)

    # Reuse correlation helper for consistent labels / codes
    corr_data = _build_corr_matrices(
        active_store=active_store,
        weights_store=weights_store,
        initial_equity=initial_equity,
    )
    if corr_data is None:
        msg = "No overlapping daily P&L data for selected strategies."
        return _empty_corr_figure(msg)

    uids = corr_data["uids"]
    labels_full = corr_data["labels_full"]
    labels_short = corr_data["labels_short"]
    codes = corr_data["codes"]

    # Build daily P&L matrix (same engine as elsewhere)
    ts_info = _build_portfolio_timeseries(
        active_store=active_store,
        weights_store=weights_store,
        weight_mode="factors",
        initial_equity=initial_equity,
    )
    if ts_info is None:
        msg = "No overlapping daily P&L data for selected strategies."
        return _empty_corr_figure(msg)

    pnl_df: pd.DataFrame = ts_info["pnl_df"]
    # Keep only the strategies used in correlation matrices, in the same order
    pnl_df = pnl_df[[uid for uid in uids if uid in pnl_df.columns]]

    # Build drawdown-overlap matrix
    ddoverlap = _build_corr6_ddoverlap_matrix(pnl_df=pnl_df, uids=uids)

    n_strat = len(uids)
    axis_labels = labels_short if n_strat <= 10 else codes

    fig = _corr_heatmap_figure(
        mat=ddoverlap,
        uids=uids,
        labels_axis=axis_labels,
        labels_full=labels_full,
        title="CORR6 – Drawdown overlap (fraction of DD days overlapping)",
        zmin=0.0,
        zmax=1.0,
        zmid=0.5,
        zfmt=".2f",
    )
    return fig


# ---------------------------------------------------------------------------
# Phase 2 – CORR7 callback (drawdown depth correlation)
# ---------------------------------------------------------------------------

@callback(
    Output("p2-corr-dddepth-heatmap", "figure"),
    Input("p1-active-list-store", "data"),
    Input("p2-weights-store", "data"),
    Input("p2-initial-equity-input", "value"),
)
def update_corr7_dddepth(
    active_store,
    weights_store,
    initial_equity,
):
    """
    Update CORR7 heatmap: correlation of drawdown depths between strategies,
    computed only on days when both are in drawdown.
    """
    active_store = active_store or []
    weights_store = weights_store or {}
    initial_equity = float(initial_equity or 100000.0)

    selected_rows = [r for r in active_store if r.get("is_selected")]
    if len(selected_rows) < 2:
        msg = "Select at least two strategies in the Active list to see drawdown depth correlation."
        return _empty_corr_figure(msg)

    corr_data = _build_corr_matrices(
        active_store=active_store,
        weights_store=weights_store,
        initial_equity=initial_equity,
    )
    if corr_data is None:
        msg = "No overlapping daily P&L data for selected strategies."
        return _empty_corr_figure(msg)

    uids = corr_data["uids"]
    labels_full = corr_data["labels_full"]
    labels_short = corr_data["labels_short"]
    codes = corr_data["codes"]

    ts_info = _build_portfolio_timeseries(
        active_store=active_store,
        weights_store=weights_store,
        weight_mode="factors",
        initial_equity=initial_equity,
    )
    if ts_info is None:
        msg = "No overlapping daily P&L data for selected strategies."
        return _empty_corr_figure(msg)

    pnl_df: pd.DataFrame = ts_info["pnl_df"]
    pnl_df = pnl_df[[uid for uid in uids if uid in pnl_df.columns]]

    mat = _build_corr7_dddepth_matrix(pnl_df=pnl_df, uids=uids)
    if mat is None or mat.empty:
        msg = "Not enough joint drawdown days to compute correlations."
        return _empty_corr_figure(msg)

    n_strat = len(uids)
    axis_labels = labels_short if n_strat <= 10 else codes

    fig = _corr_heatmap_figure(
        mat=mat,
        uids=uids,
        labels_axis=axis_labels,
        labels_full=labels_full,
        title="CORR7 – Drawdown depth correlation",
        zmin=-1.0,
        zmax=1.0,
        zmid=0.0,
        zfmt=".2f",
    )
    return fig



# ---------------------------------------------------------------------------
# Phase 2 – BETA1 callback (beta vs portfolio)
# ---------------------------------------------------------------------------

@callback(
    Output("p2-beta-portfolio-bar", "figure"),
    Output("p2-beta-summary-table", "children"),
    Input("p1-active-list-store", "data"),
    Input("p2-weights-store", "data"),
    Input("p2-initial-equity-input", "value"),
)
def update_beta_vs_portfolio(
    active_store,
    weights_store,
    initial_equity,
):
    """
    Update BETA1 bar chart and summary table: beta of each strategy's daily
    P&L vs the portfolio daily P&L (current weights).
    """
    active_store = active_store or []
    weights_store = weights_store or {}
    initial_equity = float(initial_equity or 100000.0)

    beta_df, uids, labels_full, axis_labels = _build_beta_vs_portfolio(
        active_store=active_store,
        weights_store=weights_store,
        initial_equity=initial_equity,
    )
    if beta_df is None or uids is None:
        fig = _empty_corr_figure("No data to compute betas.")
        table = _beta_summary_table_component(pd.DataFrame())
        return fig, table

    x_labels = axis_labels
    beta_vals = beta_df["beta"].astype(float).values

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=beta_vals,
            customdata=beta_df["name"].values,
            hovertemplate="Strategy: %{customdata}<br>β vs PF: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin=dict(l=40, r=20, t=40, b=60),
        title="BETA1 – Beta vs portfolio",
        xaxis=dict(title="Strategy"),
        yaxis=dict(title="β vs portfolio", zeroline=True, zerolinewidth=1),
    )

    table = _beta_summary_table_component(beta_df)
    return fig, table


# ---------------------------------------------------------------------------
# Phase 2 – SD1 callback (serial dependence)
# ---------------------------------------------------------------------------

@callback(
    Output("p2-serial-ac-bar", "figure"),
    Input("p1-active-list-store", "data"),
    Input("p2-weights-store", "data"),
    Input("p2-initial-equity-input", "value"),
)
def update_serial_ac(
    active_store,
    weights_store,
    initial_equity,
):
    """
    Update SD1 bar chart: lag-1 autocorrelation of daily P&L per strategy.
    """
    active_store = active_store or []
    weights_store = weights_store or {}
    initial_equity = float(initial_equity or 100000.0)

    ac_series, uids, labels_full, axis_labels = _build_serial_ac(
        active_store=active_store,
        weights_store=weights_store,
        initial_equity=initial_equity,
    )
    if ac_series is None or uids is None:
        return _empty_corr_figure("No data to compute serial dependence.")

    y_vals = ac_series.reindex(uids).astype(float).values
    x_labels = axis_labels

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=y_vals,
            customdata=np.array(labels_full, dtype=object),
            hovertemplate="Strategy: %{customdata}<br>AC(1): %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font={"color": "#EEEEEE"},
        margin=dict(l=40, r=20, t=40, b=60),
        title="SD1 – Serial dependence (lag-1 autocorrelation)",
        xaxis=dict(title="Strategy"),
        yaxis=dict(title="AC(1)", zeroline=True, zerolinewidth=1),
    )
    return fig



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
