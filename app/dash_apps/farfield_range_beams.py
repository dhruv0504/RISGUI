# app/dash_apps/farfield_range_beams.py
# Dash application for Far-field Range of Beams visualization

# Import Dash core components and callback utilities
from dash import Dash, html, dcc, Input, Output, State, callback_context

# Import Plotly graph objects
import plotly.graph_objects as go

# Import backend computation service and codebook exporter
from app.services.farfield_range_beams_core import compute_farfield_pattern, build_codebook_text


# Utility function to create empty placeholder figure
def empty_figure(title="", xaxis_title=None, yaxis_title=None, template="plotly_white"):

    # Print debug info
    print("Creating empty figure with title:", title)

    # Initialize empty Plotly figure
    fig = go.Figure()

    # Set title
    fig.update_layout(title=title)

    # Set axis titles if provided
    if xaxis_title or yaxis_title:
        fig.update_layout(
            xaxis_title=xaxis_title or "",
            yaxis_title=yaxis_title or ""
        )

    # Set visual theme template
    fig.update_layout(template=template)

    # Return empty figure
    return fig


# Build dynamic range input controls
def dr_control(dr_min_id: str, dr_max_id: str,
               dr_min_default: float = -30.0,
               dr_max_default: float = 0.0):

    """Return a small Div containing dynamic range (dB) min/max inputs."""

    print("Creating DR control with defaults:",
          dr_min_default, dr_max_default)

    return html.Div(
        style={"display": "grid",
               "gridTemplateColumns": "1fr 1fr",
               "gap": "8px"},
        children=[
            html.Div([
                html.Label("DR min"),
                dcc.Input(id=dr_min_id,
                          type="number",
                          value=dr_min_default,
                          style={"width": "100%"})
            ]),
            html.Div([
                html.Label("DR max"),
                dcc.Input(id=dr_max_id,
                          type="number",
                          value=dr_max_default,
                          style={"width": "100%"})
            ]),
        ],
    )


# Build RIS size control inputs
def ris_size_control(rs1_id: str, rs2_id: str,
                     rs1_default: int = 32,
                     rs2_default: int = 32):

    """Return a compact RIS size control (two numeric inputs)."""

    print("Creating RIS size control with defaults:",
          rs1_default, rs2_default)

    return html.Div(
        style={"display": "grid",
               "gridTemplateColumns": "1fr 1fr",
               "gap": "8px"},
        children=[
            html.Div([
                html.Label("RS1"),
                dcc.Input(id=rs1_id,
                          type="number",
                          value=rs1_default,
                          style={"width": "100%"})
            ]),
            html.Div([
                html.Label("RS2"),
                dcc.Input(id=rs2_id,
                          type="number",
                          value=rs2_default,
                          style={"width": "100%"})
            ]),
        ],
    )


# Build randomization toggle control
def randomization_control(rand_id: str, default: str = "On"):

    """Return a small radio control for randomization On/Off."""

    print("Creating randomization control. Default:", default)

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



# Create Dash application for Far-field Range of Beams
def create_farfield_range_beams_dash(server, url_base_pathname="/dash/far_field_range_of_beams/"):

    # Print initialization info
    print("\n========== INITIALIZING FARFIELD DASH APP ==========")
    print("URL base path:", url_base_pathname)

    # Create Dash app instance
    dash_app = Dash(
        __name__,
        server=server,
        url_base_pathname=url_base_pathname,
        suppress_callback_exceptions=True
    )

    # Set browser tab title
    dash_app.title = "Far-field Range of Beams (Dash)"

    # Define entire UI layout
    dash_app.layout = html.Div(

        # Global page style
        style={"fontFamily": "Arial", "padding": "14px"},

        children=[

            # Page header
            html.H2("Far-field Range of Beams (RIS)"),

            # Subtitle / information text
            html.Div("RIS Size default: 32 by 32 (changeable)",
                     style={"marginBottom": "10px"}),

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

                            # Incidence angle section
                            html.H4("Angle of Incidence (deg)"),

                            # Grid layout for θi and φi
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "1fr 1fr",
                                    "gap": "8px"
                                },

                                children=[

                                    # θi input
                                    html.Div([
                                        html.Label("θi (deg)"),
                                        dcc.Input(
                                            id="ti",
                                            type="number",
                                            value=0.0,
                                            style={"width":"100%"}
                                        )
                                    ]),

                                    # φi input
                                    html.Div([
                                        html.Label("φi (deg)"),
                                        dcc.Input(
                                            id="pi",
                                            type="number",
                                            value=0.0,
                                            style={"width":"100%"}
                                        )
                                    ]),
                                ]
                            ),

                            html.Hr(),

                            # Reflection angle section
                            html.H4("Angles of Reflection (deg)"),

                            # Grid layout for sweep parameters
                            html.Div(
                                style={
                                    "display":"grid",
                                    "gridTemplateColumns":"70px 70px 70px 70px",
                                    "gap":"6px"
                                },

                                children=[

                                    # θr start
                                    html.Div([
                                        html.Label("θr start"),
                                        dcc.Input(
                                            id="trstart",
                                            type="number",
                                            value=0.0,
                                            style={"width":"100%"}
                                        )
                                    ]),

                                    # θr step
                                    html.Div([
                                        html.Label("θr step"),
                                        dcc.Input(
                                            id="trstep",
                                            type="number",
                                            value=1.0,
                                            style={"width":"100%"}
                                        )
                                    ]),

                                    # θr stop
                                    html.Div([
                                        html.Label("θr stop"),
                                        dcc.Input(
                                            id="trstop",
                                            type="number",
                                            value=0.0,
                                            style={"width":"100%"}
                                        )
                                    ]),

                                    # Spacer
                                    html.Div([]),

                                    # φr start
                                    html.Div([
                                        html.Label("φr start"),
                                        dcc.Input(
                                            id="prstart",
                                            type="number",
                                            value=30.0,
                                            style={"width":"100%"}
                                        )
                                    ]),

                                    # φr step
                                    html.Div([
                                        html.Label("φr step"),
                                        dcc.Input(
                                            id="prstep",
                                            type="number",
                                            value=1.0,
                                            style={"width":"100%"}
                                        )
                                    ]),

                                    # φr stop
                                    html.Div([
                                        html.Label("φr stop"),
                                        dcc.Input(
                                            id="prstop",
                                            type="number",
                                            value=90.0,
                                            style={"width":"100%"}
                                        )
                                    ]),
                                ]
                            ),

                            html.Br(),

                            # Randomization toggle
                            randomization_control('rand', default='Off'),

                            html.Hr(),

                            # Dynamic range section
                            html.H4("Dynamic Range (dB)"),

                            # DR input control
                            dr_control(
                                'dr_min',
                                'dr_max',
                                dr_min_default=-30,
                                dr_max_default=0
                            ),

                            html.Hr(),

                            # RIS size section
                            html.H4("RIS Size"),

                            # RIS size control
                            ris_size_control(
                                'rs1',
                                'rs2',
                                rs1_default=32,
                                rs2_default=32
                            ),

                            html.Hr(),

                            # Buttons row
                            html.Div(
                                style={"display":"flex","gap":"10px"},

                                children=[

                                    # Update button
                                    html.Button(
                                        "Update",
                                        id="btn_update",
                                        n_clicks=0,
                                        style={"padding":"8px 12px"}
                                    ),

                                    # Clear button
                                    html.Button(
                                        "Clear Plot",
                                        id="btn_clear",
                                        n_clicks=0,
                                        style={"padding":"8px 12px"}
                                    ),

                                    # Download component
                                    dcc.Download(id="download_codebook"),
                                ]
                            ),

                            # Status message area
                            html.Div(
                                id="status",
                                style={"marginTop":"10px","color":"#444"}
                            )
                        ]
                    ),

                    # ================= RIGHT PLOT PANEL =================
                    html.Div(
                        style={
                            "border": "1px solid #ddd",
                            "borderRadius": "10px",
                            "padding": "12px"
                        },

                        children=[

                            # Beam pattern graph
                            dcc.Graph(
                                id="beam_plot",
                                style={"height":"78vh"}
                            ),

                            # Hidden storage for codebook
                            dcc.Store(id="stored_codebook"),
                        ],
                    ),
                ],
            ),
        ],
    )

    print("Dash layout constructed successfully.")

        # ================= MAIN UPDATE CALLBACK =================
    @dash_app.callback(
        # Output: updated figure
        Output("beam_plot", "figure"),

        # Output: stored codebook data
        Output("stored_codebook", "data"),

        # Output: status text
        Output("status", "children"),

        # Trigger inputs
        Input("btn_update", "n_clicks"),
        Input("btn_clear", "n_clicks"),

        # State inputs (form values)
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

    # Callback function that updates plot
    def update_plot(n_update, n_clear,
                    ti, pi,
                    prstart, prstep, prstop,
                    trstart, trstep, trstop,
                    rand_value, rs1, rs2, dr_min, dr_max):

        print("\n========== UPDATE PLOT CALLBACK TRIGGERED ==========")

        # Determine which input triggered callback
        triggered = callback_context.triggered

        # Create empty default figure
        empty_fig = empty_figure(
            title="Beam pattern",
            xaxis_title="u",
            yaxis_title="v"
        )

        # If nothing triggered yet (initial load)
        if not triggered:
            print("No trigger detected. Returning cleared state.")
            return empty_fig, None, "Cleared."

        # Extract triggering property
        prop = triggered[0]["prop_id"].split(".")[0]

        print("Triggered by:", prop)

        # Handle clear button
        if prop == "btn_clear" or (prop == "" and n_update == 0):

            print("Clear action executed.")
            return empty_fig, None, "Cleared."

        # Build parameters dictionary for backend
        params = {
            "ti": ti,
            "pi": pi,
            "prstart": prstart,
            "prstep": prstep,
            "prstop": prstop,
            "trstart": trstart,
            "trstep": trstep,
            "trstop": trstop,
            "rand": rand_value,
            "rs1": rs1,
            "rs2": rs2,
            "dr_min": dr_min,
            "dr_max": dr_max
        }

        print("Calling backend with parameters:")
        print(params)

        # Call backend computation
        result = compute_farfield_pattern(params)

        # Extract figure
        fig = result.get("figure")

        # Extract codebook
        codebook = result.get("codebook")

        # Extract status
        status = result.get("status", "")

        print("Backend status:", status)
        print("========== UPDATE CALLBACK END ==========\n")

        # Return outputs
        return fig, codebook, status


    # ================= DOWNLOAD CALLBACK =================
    @dash_app.callback(
        Output("download_codebook", "data"),
        Input("btn_download", "n_clicks"),
        State("stored_codebook", "data"),
        prevent_initial_call=True
    )

    # Function to handle codebook download
    def download_codebook(n_clicks, stored_codebook):

        print("\nDownload codebook callback triggered.")

        # If no stored codebook available
        if not stored_codebook:

            print("No codebook stored. Download aborted.")
            return dash.no_update

        # Convert stored list to numpy array
        import numpy as _np
        bits = _np.array(stored_codebook, dtype=int)

        print("Codebook shape for download:", bits.shape)

        # Convert to text format
        txt = build_codebook_text(bits)

        print("Download file prepared.")

        # Return downloadable file
        return dict(
            content=txt,
            filename="farfield_codebook.txt",
            type="text/plain"
        )

    # Print successful initialization
    print("========== FARFIELD DASH APP READY ==========\n")

    # Return Dash app instance
    return dash_app