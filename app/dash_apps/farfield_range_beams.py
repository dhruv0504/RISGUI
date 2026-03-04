# app/dash_apps/farfield_range_beams.py
from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.graph_objects as go
from app.services.farfield_range_beams_core import compute_farfield_pattern, build_codebook_text


def empty_figure(title="", xaxis_title=None, yaxis_title=None, template="plotly_white"):
    fig = go.Figure()
    fig.update_layout(title=title)
    if xaxis_title or yaxis_title:
        fig.update_layout(xaxis_title=xaxis_title or "", yaxis_title=yaxis_title or "")
    fig.update_layout(template=template)
    return fig


def dr_control(dr_min_id: str, dr_max_id: str, dr_min_default: float = -30.0, dr_max_default: float = 0.0):
    """Return a small Div containing dynamic range (dB) min/max inputs."""
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
        children=[
            html.Div([html.Label("DR min"), dcc.Input(id=dr_min_id, type="number", value=dr_min_default, style={"width": "100%"})]),
            html.Div([html.Label("DR max"), dcc.Input(id=dr_max_id, type="number", value=dr_max_default, style={"width": "100%"})]),
        ],
    )


def ris_size_control(rs1_id: str, rs2_id: str, rs1_default: int = 32, rs2_default: int = 32):
    """Return a compact RIS size control (two numeric inputs)."""
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
        children=[
            html.Div([html.Label("RS1"), dcc.Input(id=rs1_id, type="number", value=rs1_default, style={"width": "100%"})]),
            html.Div([html.Label("RS2"), dcc.Input(id=rs2_id, type="number", value=rs2_default, style={"width": "100%"})]),
        ],
    )


def randomization_control(rand_id: str, default: str = "On"):
    """Return a small radio control for randomization On/Off."""
    return html.Div([
        html.Label("Randomization"),
        dcc.RadioItems(id=rand_id, options=[{"label": "On", "value": "On"}, {"label": "Off", "value": "Off"}], value=default, inline=True),
    ], style={"marginTop": "6px"})

def create_farfield_range_beams_dash(server, url_base_pathname="/dash/far_field_range_of_beams/"):
    dash_app = Dash(__name__, server=server, url_base_pathname=url_base_pathname, suppress_callback_exceptions=True)
    dash_app.title = "Far-field Range of Beams (Dash)"

    dash_app.layout = html.Div(
        style={"fontFamily": "Arial", "padding": "14px"},
        children=[
            html.H2("Far-field Range of Beams (RIS)"),
            html.Div("RIS Size default: 32 by 32 (changeable)", style={"marginBottom": "10px"}),

            html.Div(
                style={"display": "grid", "gridTemplateColumns": "360px 1fr", "gap": "14px"},
                children=[
                    # Controls (left)
                    html.Div(
                        style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px"},
                        children=[
                            html.H4("Angle of Incidence (deg)"),
                            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"8px"}, children=[
                                html.Div([html.Label("θi (deg)"), dcc.Input(id="ti", type="number", value=0.0, style={"width":"100%"})]),
                                html.Div([html.Label("φi (deg)"), dcc.Input(id="pi", type="number", value=0.0, style={"width":"100%"})]),
                            ]),
                            html.Hr(),
                            html.H4("Angles of Reflection (deg)"),
                            html.Div(style={"display":"grid","gridTemplateColumns":"70px 70px 70px 70px","gap":"6px"}, children=[
                                html.Div([html.Label("θr start"), dcc.Input(id="trstart", type="number", value=0.0, style={"width":"100%"})]),
                                html.Div([html.Label("θr step"),  dcc.Input(id="trstep",  type="number", value=1.0, style={"width":"100%"})]),
                                html.Div([html.Label("θr stop"),  dcc.Input(id="trstop",  type="number", value=0.0, style={"width":"100%"})]),
                                html.Div([]),
                                html.Div([html.Label("φr start"), dcc.Input(id="prstart", type="number", value=30.0, style={"width":"100%"})]),
                                html.Div([html.Label("φr step"),  dcc.Input(id="prstep",  type="number", value=1.0, style={"width":"100%"})]),
                                html.Div([html.Label("φr stop"),  dcc.Input(id="prstop",  type="number", value=90.0, style={"width":"100%"})]),
                            ]),
                            html.Br(),
                                randomization_control('rand', default='Off'),
                            html.Hr(),
                            html.H4("Dynamic Range (dB)"),
                            dr_control('dr_min', 'dr_max', dr_min_default=-30, dr_max_default=0),
                            html.Hr(),
                            html.H4("RIS Size"),
                            ris_size_control('rs1', 'rs2', rs1_default=32, rs2_default=32),
                            html.Hr(),
                            html.Div(style={"display":"flex","gap":"10px"}, children=[
                                html.Button("Update", id="btn_update", n_clicks=0, style={"padding":"8px 12px"}),
                                html.Button("Clear Plot", id="btn_clear", n_clicks=0, style={"padding":"8px 12px"}),
                                # html.Button("Download Codebook", id="btn_download", n_clicks=0, style={"padding":"8px 12px"}),
                                dcc.Download(id="download_codebook"),
                            ]),
                            html.Div(id="status", style={"marginTop":"10px","color":"#444"})
                        ]
                    ),

                    # Plot (right)
                    html.Div(
                        style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px"},
                        children=[
                            dcc.Graph(id="beam_plot", style={"height":"78vh"}),
                            dcc.Store(id="stored_codebook"),
                        ],
                    ),
                ],
            ),
        ],
    )

    # Callbacks very similar to your nearfield file
    @dash_app.callback(
        Output("beam_plot", "figure"),
        Output("stored_codebook", "data"),
        Output("status", "children"),
        Input("btn_update", "n_clicks"),
        Input("btn_clear", "n_clicks"),
        State("ti", "value"),
        State("pi", "value"),
        State("prstart", "value"),
        State("prstep", "value"),
        State("prstop", "value"),
        State("trstart", "value"),
        State("trstep", "value"),
        State("trstop", "value"),
        State("rand", "value"),
        State("rs1", "value"),
        State("rs2", "value"),
        State("dr_min", "value"),
        State("dr_max", "value"),
    )
    def update_plot(n_update, n_clear,
                    ti, pi,
                    prstart, prstep, prstop,
                    trstart, trstep, trstop,
                    rand_value, rs1, rs2, dr_min, dr_max):
        triggered = callback_context.triggered
        empty_fig = empty_figure(title="Beam pattern", xaxis_title="u", yaxis_title="v")
        if not triggered:
            return empty_fig, None, "Cleared."

        prop = triggered[0]["prop_id"].split(".")[0]
        if prop == "btn_clear" or (prop == "" and n_update == 0):
            return empty_fig, None, "Cleared."

        params = {
            "ti": ti, "pi": pi,
            "prstart": prstart, "prstep": prstep, "prstop": prstop,
            "trstart": trstart, "trstep": trstep, "trstop": trstop,
            "rand": rand_value,
            "rs1": rs1, "rs2": rs2,
            "dr_min": dr_min, "dr_max": dr_max
        }

        result = compute_farfield_pattern(params)
        fig = result.get("figure")
        codebook = result.get("codebook")
        status = result.get("status", "")

        return fig, codebook, status

    @dash_app.callback(
        Output("download_codebook", "data"),
        Input("btn_download", "n_clicks"),
        State("stored_codebook", "data"),
        prevent_initial_call=True
    )
    def download_codebook(n_clicks, stored_codebook):
        if not stored_codebook:
            return dash.no_update

        import numpy as _np
        bits = _np.array(stored_codebook, dtype=int)
        txt = build_codebook_text(bits)
        return dict(content=txt, filename="farfield_codebook.txt", type="text/plain")

    return dash_app
