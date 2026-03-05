# app/dash_apps/nearfield_range_beams.py
# Dash application for Near-field Range of Beams visualization

# Import Dash core components and callback utilities
from dash import Dash, html, dcc, Input, Output, State, callback_context

# Import Plotly graph objects for visualization
import plotly.graph_objects as go


# Utility function to build an empty placeholder figure
def empty_figure(title="", xaxis_title=None, yaxis_title=None, template="plotly_white"):

    # Print debug info
    print("Creating empty figure with title:", title)

    # Create empty Plotly figure
    fig = go.Figure()

    # Set figure title
    fig.update_layout(title=title)

    # Optionally set axis titles
    if xaxis_title or yaxis_title:
        fig.update_layout(
            xaxis_title=xaxis_title or "",
            yaxis_title=yaxis_title or ""
        )

    # Apply visual template
    fig.update_layout(template=template)

    # Return figure
    return fig


# Build Dynamic Range input controls
def dr_control(dr_min_id: str, dr_max_id: str,
               dr_min_default: float = -30.0,
               dr_max_default: float = 0.0):

    """Return a small Div containing dynamic range (dB) min/max inputs."""

    print("Building DR control with defaults:",
          dr_min_default, dr_max_default)

    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "8px"
        },
        children=[
            html.Div([
                html.Label("DR min"),
                dcc.Input(
                    id=dr_min_id,
                    type="number",
                    value=dr_min_default,
                    style={"width": "100%"}
                )
            ]),
            html.Div([
                html.Label("DR max"),
                dcc.Input(
                    id=dr_max_id,
                    type="number",
                    value=dr_max_default,
                    style={"width": "100%"}
                )
            ]),
        ],
    )


# Build randomization toggle control
def randomization_control(rand_id: str, default: str = "On"):

    """Return a small radio control for randomization On/Off."""

    print("Building randomization control. Default:", default)

    return html.Div([
        html.Label("Randomization"),
        dcc.RadioItems(
            id=rand_id,
            options=[
                {"label": "On", "value": "On"},
                {"label": "Off", "value": "Off"}
            ],
            value=default,
            inline=True
        ),
    ], style={"marginTop": "6px"})

# Import backend computation service
from app.services.nearfield_range_beams_core import compute_nearfield_pattern


# Create Dash application for Near-field Range of Beams
def create_nearfield_range_beams_dash(server,
                                      url_base_pathname="/dash/near_field_range_of_beams/"):

    # Print initialization banner
    print("\n========== INITIALIZING NEARFIELD RANGE DASH ==========")
    print("URL base path:", url_base_pathname)

    # Create Dash app instance
    dash_app = Dash(
        __name__,
        server=server,
        url_base_pathname=url_base_pathname,
        suppress_callback_exceptions=True
    )

    # Set browser tab title
    dash_app.title = "Near-field Range of Beams (Dash)"

    # Define full layout
    dash_app.layout = html.Div(

        # Global styling
        style={"fontFamily": "Arial", "padding": "14px"},

        children=[

            # Page header
            html.H2("Near-field Range of Beams (RIS)"),

            # Subtitle text
            html.Div(
                "RIS Size is fixed: 32 by 32",
                style={"marginBottom": "10px"}
            ),

            # Main layout grid (controls left, plot right)
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "360px 1fr",
                    "gap": "14px"
                },

                children=[

                    # ================= LEFT CONTROL PANEL =================
                    html.Div(
                        style={
                            "border": "1px solid #ddd",
                            "borderRadius": "10px",
                            "padding": "12px"
                        },

                        children=[

                            # Transmitter section
                            html.H4("Transmitter (Tx) Position (mm)"),

                            # Grid for xi, yi, zi inputs
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "1fr 1fr 1fr",
                                    "gap": "8px"
                                },

                                children=[

                                    # xi input
                                    html.Div([
                                        html.Label("xᵢ"),
                                        dcc.Input(
                                            id="xi",
                                            type="number",
                                            value=0.0,
                                            style={"width": "100%"}
                                        )
                                    ]),

                                    # yi input
                                    html.Div([
                                        html.Label("yᵢ"),
                                        dcc.Input(
                                            id="yi",
                                            type="number",
                                            value=145.0,
                                            style={"width": "100%"}
                                        )
                                    ]),

                                    # zi input
                                    html.Div([
                                        html.Label("zᵢ"),
                                        dcc.Input(
                                            id="zi",
                                            type="number",
                                            value=250.0,
                                            style={"width": "100%"}
                                        )
                                    ]),
                                ]
                            ),

                            html.Hr(),

                            # Receiver / Focus section
                            html.H4("Receiver / Focus Plane"),

                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "1fr 1fr",
                                    "gap": "8px"
                                },

                                children=[

                                    # Zr input
                                    html.Div([
                                        html.Label("Zᵣ (mm)"),
                                        dcc.Input(
                                            id="zr",
                                            type="number",
                                            value=300.0,
                                            style={"width": "100%"}
                                        )
                                    ]),

                                    # Randomization control
                                    randomization_control('rand', default='On'),
                                ]
                            ),

                            html.Hr(),

                            # Focus sweep section
                            html.H4("Focus Sweep (xᵣ, yᵣ) (mm)"),

                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "70px 1fr 1fr 1fr",
                                    "gap": "6px"
                                },

                                children=[

                                    # Header row
                                    html.Div([]),
                                    html.Div(html.B("Start")),
                                    html.Div(html.B("Step")),
                                    html.Div(html.B("Stop")),

                                    # x_r row
                                    html.Div([html.Label("xᵣ")]),
                                    html.Div(dcc.Input(
                                        id="x_start",
                                        type="number",
                                        value=0.0,
                                        style={"width": "100%"}
                                    )),
                                    html.Div(dcc.Input(
                                        id="x_step",
                                        type="number",
                                        value=10.0,
                                        style={"width": "100%"}
                                    )),
                                    html.Div(dcc.Input(
                                        id="x_stop",
                                        type="number",
                                        value=0.0,
                                        style={"width": "100%"}
                                    )),

                                    # y_r row
                                    html.Div([html.Label("yᵣ")]),
                                    html.Div(dcc.Input(
                                        id="y_start",
                                        type="number",
                                        value=0.0,
                                        style={"width": "100%"}
                                    )),
                                    html.Div(dcc.Input(
                                        id="y_step",
                                        type="number",
                                        value=10.0,
                                        style={"width": "100%"}
                                    )),
                                    html.Div(dcc.Input(
                                        id="y_stop",
                                        type="number",
                                        value=0.0,
                                        style={"width": "100%"}
                                    )),
                                ]
                            ),

                            html.Hr(),

                            # Scan grid section
                            html.H4("Scan Grid (FoV)"),

                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "1fr 1fr",
                                    "gap": "8px"
                                },

                                children=[

                                    html.Div([
                                        html.Label("x_range (mm)"),
                                        dcc.Input(
                                            id="x_range",
                                            type="number",
                                            value=1000,
                                            style={"width": "100%"}
                                        )
                                    ]),

                                    html.Div([
                                        html.Label("x_step (mm)"),
                                        dcc.Input(
                                            id="x_scan_step",
                                            type="number",
                                            value=50,
                                            style={"width": "100%"}
                                        )
                                    ]),

                                    html.Div([
                                        html.Label("y_range (mm)"),
                                        dcc.Input(
                                            id="y_range",
                                            type="number",
                                            value=1000,
                                            style={"width": "100%"}
                                        )
                                    ]),

                                    html.Div([
                                        html.Label("y_step (mm)"),
                                        dcc.Input(
                                            id="y_scan_step",
                                            type="number",
                                            value=50,
                                            style={"width": "100%"}
                                        )
                                    ]),
                                ]
                            ),

                            html.Hr(),

                            # Dynamic range controls
                            html.H4("Dynamic Range (dB)"),
                            dr_control('dr_min', 'dr_max',
                                       dr_min_default=-30,
                                       dr_max_default=0),

                            html.Hr(),

                            # Buttons
                            html.Div(
                                style={"display": "flex", "gap": "10px"},
                                children=[

                                    html.Button(
                                        "Update",
                                        id="btn_update",
                                        n_clicks=0,
                                        style={"padding": "8px 12px"}
                                    ),

                                    html.Button(
                                        "Clear Plot",
                                        id="btn_clear",
                                        n_clicks=0,
                                        style={"padding": "8px 12px"}
                                    ),

                                    html.Button(
                                        "Download Codebook",
                                        id="btn_download",
                                        n_clicks=0,
                                        style={"padding": "8px 12px"}
                                    ),

                                    dcc.Download(id="download_codebook"),
                                ]
                            ),

                            # Status message
                            html.Div(
                                id="status",
                                style={
                                    "marginTop": "10px",
                                    "color": "#444"
                                }
                            )
                        ],
                    ),

                    # ================= RIGHT PLOT PANEL =================
                    html.Div(
                        style={
                            "border": "1px solid #ddd",
                            "borderRadius": "10px",
                            "padding": "12px"
                        },

                        children=[

                            # Main beam plot
                            dcc.Graph(
                                id="beam_plot",
                                style={"height": "78vh"}
                            ),

                            # Hidden storage for codebook
                            dcc.Store(id="stored_codebook"),
                        ],
                    ),
                ],
            ),
        ],
    )

    print("Nearfield layout constructed successfully.")
    # ================= MAIN UPDATE CALLBACK =================
    @dash_app.callback(
        # Output updated beam figure
        Output("beam_plot", "figure"),

        # Output stored codebook
        Output("stored_codebook", "data"),

        # Output status message
        Output("status", "children"),

        # Trigger inputs
        Input("btn_update", "n_clicks"),
        Input("btn_clear", "n_clicks"),

        # State inputs (form values)
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

    # Callback function definition
    def update_plot(
        n_update, n_clear,
        xi, yi, zi, zr,
        rand_value,
        x_start, x_step, x_stop,
        y_start, y_step, y_stop,
        x_range, y_range, x_scan_step, y_scan_step,
        dr_min, dr_max
    ):

        print("\n========== NEARFIELD UPDATE CALLBACK ==========")

        # Determine which input triggered callback
        triggered = callback_context.triggered

        # Create empty default figure
        empty_fig = empty_figure(
            title="Beam pattern",
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)"
        )

        # If nothing triggered (initial load)
        if not triggered:
            print("No trigger detected. Returning cleared state.")
            return empty_fig, None, "Cleared."

        # Extract triggering property
        prop = triggered[0]["prop_id"].split(".")[0]

        print("Triggered by:", prop)

        # If clear button pressed
        if prop == "btn_clear" or (prop == "" and n_update == 0):
            print("Clear action executed.")
            return empty_fig, None, "Cleared."

        # Build parameter dictionary for backend
        params = {
            "xi": xi,
            "yi": yi,
            "zi": zi,
            "zr": zr,
            "rand": rand_value,
            "x_start": x_start,
            "x_step": x_step,
            "x_stop": x_stop,
            "y_start": y_start,
            "y_step": y_step,
            "y_stop": y_stop,
            "x_range": x_range,
            "y_range": y_range,
            "x_scan_step": x_scan_step,
            "y_scan_step": y_scan_step,
            "dr_min": dr_min,
            "dr_max": dr_max
        }

        print("Calling backend with parameters:")
        print(params)

        # Call backend nearfield computation
        result = compute_nearfield_pattern(params)

        # Extract figure from result
        fig = result.get("figure")

        # Extract codebook
        codebook = result.get("codebook")

        # Extract status message
        status = result.get("status", "")

        print("Backend status:", status)
        print("========== NEARFIELD UPDATE END ==========\n")

        # Return updated outputs
        return fig, codebook, status


    # ================= DOWNLOAD CALLBACK =================
    @dash_app.callback(
        Output("download_codebook", "data"),
        Input("btn_download", "n_clicks"),
        State("stored_codebook", "data"),
        prevent_initial_call=True
    )

    # Callback function for downloading codebook
    def download_codebook(n_clicks, stored_codebook):

        print("\nDownload codebook callback triggered.")

        # If no codebook stored
        if not stored_codebook:
            print("No stored codebook. Download aborted.")
            return dash.no_update

        # Import backend exporter
        from app.services.nearfield_range_beams_core import build_codebook_text

        # Import numpy
        import numpy as np

        # Convert stored list to numpy array
        codebook_bits = np.array(stored_codebook, dtype=int)  # (N,32,32)

        print("Codebook shape for download:", codebook_bits.shape)

        # Convert to text
        txt = build_codebook_text(codebook_bits)

        print("Download file prepared.")

        # Return downloadable file object
        return dict(
            content=txt,
            filename="codebook.txt",
            type="text/plain"
        )

    # Print successful initialization
    print("========== NEARFIELD DASH READY ==========\n")

    # Return Dash application
    return dash_app
