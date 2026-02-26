# app/dash_apps/nearfield_range_beams.py
from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.graph_objects as go
from app.dash_apps.callback_helpers import empty_figure
from app.dash_apps.ui_helpers import dr_control, randomization_control
from app.services.nearfield_range_beams_core import compute_nearfield_pattern

def create_nearfield_range_beams_dash(server, url_base_pathname="/dash/near_field_range_of_beams/"):
    dash_app = Dash(__name__, server=server, url_base_pathname=url_base_pathname, suppress_callback_exceptions=True)
    dash_app.title = "Near-field Range of Beams (Dash)"

    dash_app.layout = html.Div(
        style={"fontFamily": "Arial", "padding": "14px"},
        children=[
            html.H2("Near-field Range of Beams (RIS)"),
            html.Div("RIS Size is fixed: 32 by 32", style={"marginBottom": "10px"}),

            html.Div(
                style={"display": "grid", "gridTemplateColumns": "360px 1fr", "gap": "14px"},
                children=[
                    # Controls (left)
                    html.Div(
                        style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px"},
                        children=[
                            html.H4("Transmitter (Tx) Position (mm)"),
                            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "8px"}, children=[
                                html.Div([html.Label("xᵢ"), dcc.Input(id="xi", type="number", value=0.0, style={"width": "100%"})]),
                                html.Div([html.Label("yᵢ"), dcc.Input(id="yi", type="number", value=145.0, style={"width": "100%"})]),
                                html.Div([html.Label("zᵢ"), dcc.Input(id="zi", type="number", value=250.0, style={"width": "100%"})]),
                            ]),

                            html.Hr(),
                            html.H4("Receiver / Focus Plane"),
                            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"}, children=[
                                html.Div([html.Label("Zᵣ (mm)"), dcc.Input(id="zr", type="number", value=300.0, style={"width": "100%"})]),
                                randomization_control('rand', default='On'),
                            ]),

                            html.Hr(),
                            html.H4("Focus Sweep (xᵣ, yᵣ) (mm)"),
                            html.Div(style={"display": "grid", "gridTemplateColumns": "70px 1fr 1fr 1fr", "gap": "6px"}, children=[
                                html.Div([]), html.Div(html.B("Start")), html.Div(html.B("Step")), html.Div(html.B("Stop")),
                                html.Div([html.Label("xᵣ")]),
                                html.Div(dcc.Input(id="x_start", type="number", value=0.0, style={"width": "100%"})),
                                html.Div(dcc.Input(id="x_step", type="number", value=10.0, style={"width": "100%"})),
                                html.Div(dcc.Input(id="x_stop", type="number", value=0.0, style={"width": "100%"})),

                                html.Div([html.Label("yᵣ")]),
                                html.Div(dcc.Input(id="y_start", type="number", value=0.0, style={"width": "100%"})),
                                html.Div(dcc.Input(id="y_step", type="number", value=10.0, style={"width": "100%"})),
                                html.Div(dcc.Input(id="y_stop", type="number", value=0.0, style={"width": "100%"})),
                            ]),

                            html.Hr(),
                            html.H4("Scan Grid (FoV)"),
                            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"}, children=[
                                html.Div([html.Label("x_range (mm)"), dcc.Input(id="x_range", type="number", value=1000, style={"width": "100%"})]),
                                html.Div([html.Label("x_step (mm)"), dcc.Input(id="x_scan_step", type="number", value=50, style={"width": "100%"})]),
                                html.Div([html.Label("y_range (mm)"), dcc.Input(id="y_range", type="number", value=1000, style={"width": "100%"})]),
                                html.Div([html.Label("y_step (mm)"), dcc.Input(id="y_scan_step", type="number", value=50, style={"width": "100%"})]),
                            ]),

                            html.Hr(),
                            html.H4("Dynamic Range (dB)"),
                            dr_control('dr_min', 'dr_max', dr_min_default=-30, dr_max_default=0),

                            html.Hr(),
                            html.Div(style={"display": "flex", "gap": "10px"}, children=[
                                html.Button("Update", id="btn_update", n_clicks=0, style={"padding": "8px 12px"}),
                                html.Button("Clear Plot", id="btn_clear", n_clicks=0, style={"padding": "8px 12px"}),
                                html.Button("Download Codebook", id="btn_download", n_clicks=0, style={"padding": "8px 12px"}),
                                dcc.Download(id="download_codebook"),
                            ]),
                            html.Div(id="status", style={"marginTop": "10px", "color": "#444"})
                        ],
                    ),

                    # Plot (right)
                    html.Div(
                        style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px"},
                        children=[
                            dcc.Graph(id="beam_plot", style={"height": "78vh"}),
                            dcc.Store(id="stored_codebook"),
                        ],
                    ),
                ],
            ),
        ],
    )

    # Callbacks
    @dash_app.callback(
        Output("beam_plot", "figure"),
        Output("stored_codebook", "data"),
        Output("status", "children"),
        Input("btn_update", "n_clicks"),
        Input("btn_clear", "n_clicks"),
        State("xi", "value"),
        State("yi", "value"),
        State("zi", "value"),
        State("zr", "value"),
        State("rand", "value"),
        State("x_start", "value"),
        State("x_step", "value"),
        State("x_stop", "value"),
        State("y_start", "value"),
        State("y_step", "value"),
        State("y_stop", "value"),
        State("x_range", "value"),
        State("y_range", "value"),
        State("x_scan_step", "value"),
        State("y_scan_step", "value"),
        State("dr_min", "value"),
        State("dr_max", "value"),
    )
    def update_plot(
        n_update, n_clear,
        xi, yi, zi, zr,
        rand_value,
        x_start, x_step, x_stop,
        y_start, y_step, y_stop,
        x_range, y_range, x_scan_step, y_scan_step,
        dr_min, dr_max
    ):
        triggered = callback_context.triggered
        empty_fig = empty_figure(title="Beam pattern", xaxis_title="X (mm)", yaxis_title="Y (mm)")

        if not triggered:
            return empty_fig, None, "Cleared."

        prop = triggered[0]["prop_id"].split(".")[0]
        if prop == "btn_clear" or (prop == "" and n_update == 0):
            return empty_fig, None, "Cleared."

        params = {
            "xi": xi, "yi": yi, "zi": zi, "zr": zr, "rand": rand_value,
            "x_start": x_start, "x_step": x_step, "x_stop": x_stop,
            "y_start": y_start, "y_step": y_step, "y_stop": y_stop,
            "x_range": x_range, "y_range": y_range,
            "x_scan_step": x_scan_step, "y_scan_step": y_scan_step,
            "dr_min": dr_min, "dr_max": dr_max
        }

        result = compute_nearfield_pattern(params)
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

        from app.services.nearfield_range_beams_core import build_codebook_text
        import numpy as np

        codebook_bits = np.array(stored_codebook, dtype=int)  # (N,32,32)
        txt = build_codebook_text(codebook_bits)

        return dict(
            content=txt,
            filename="codebook.txt",
            type="text/plain"
        )

    return dash_app
