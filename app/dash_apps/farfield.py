# app/dash_apps/farfield.py
# Main Dash application for Far-field Reflectarray Visualizer

# Used to print full stack traces if runtime errors occur
import traceback

# Import Dash core classes and callback tools
from dash import Dash, html, dcc, Input, Output, State

# Import Dash namespace (used for dash.no_update and callback_context)
import dash

# Import Plotly graph objects for figure creation
import plotly.graph_objects as go


# Helper function to sync a slider and numeric input
def register_sync_pair(app, slider_id, input_id, minv, maxv, cast=int):

    """Register a sync callback between a slider and numeric input.

    Ensures the two controls remain in sync and clamps input to [minv, maxv].
    """

    print(f"Registering sync pair: {slider_id} <-> {input_id}")

    # Create callback linking slider and input
    @app.callback(
        Output(slider_id, "value"),   # Update slider value
        Output(input_id, "value"),    # Update numeric input value
        Input(slider_id, "value"),    # Triggered when slider changes
        Input(input_id, "value"),     # Triggered when input changes
    )
    def _sync(slider_val, input_val):

        # Identify which component triggered callback
        triggered_id = dash.callback_context.triggered_id

        # If slider changed, mirror value to input
        if triggered_id == slider_id:
            return slider_val, slider_val

        # If numeric input changed
        if triggered_id == input_id:

            # If input empty
            if input_val is None:
                return dash.no_update, dash.no_update

            # Attempt type conversion
            try:
                v = cast(input_val)
            except Exception:
                return dash.no_update, dash.no_update

            # Clamp to valid range
            v = max(minv, min(maxv, v))

            return v, v

        # Otherwise no update
        return dash.no_update, dash.no_update


# Utility to build empty placeholder figure
def empty_figure(title="", xaxis_title=None, yaxis_title=None, template="plotly_white"):

    print("Creating empty placeholder figure.")

    fig = go.Figure()

    fig.update_layout(title=title)

    if xaxis_title or yaxis_title:
        fig.update_layout(
            xaxis_title=xaxis_title or "",
            yaxis_title=yaxis_title or ""
        )

    fig.update_layout(template=template)

    return fig


# Build Dynamic Range input controls
def dr_control(dr_min_id: str, dr_max_id: str,
               dr_min_default: float = -30.0,
               dr_max_default: float = 0.0):

    """Return a small Div containing dynamic range (dB) min/max inputs."""

    print("Building DR control with defaults:",
          dr_min_default, dr_max_default)

    return html.Div(
        style={"display": "grid",
               "gridTemplateColumns": "1fr 1fr",
               "gap": "8px"},
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


# Build randomization toggle
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

# Import backend computation functions and constants
from app.services.farfield_core import (
    compute_fields,               # Core electromagnetic computation
    build_vector_figure_visible,  # 3D radiation surface builder
    build_phase_figure,           # Phase mask visualization
    build_beam_figure,            # 2D beam projection visualization
    STATIC_RS1,                   # Fixed RIS X dimension
    STATIC_RS2                    # Fixed RIS Y dimension
)


# Create Dash application for Far-field visualization
def create_farfield_dash(server, url_base_pathname="/dash/far_field/"):

    print("\n========== INITIALIZING FARFIELD DASH ==========")
    print("URL base path:", url_base_pathname)

    # Create Dash app instance
    dash_app = Dash(
        __name__,
        server=server,
        url_base_pathname=url_base_pathname,
        suppress_callback_exceptions=True
    )

    # Attempt to build initial figures safely
    try:

        print("Building initial figures...")

        # Run initial computation with default angles
        init_data = compute_fields(
            30,   # θi
            90,   # φi
            90,   # θr
            150,  # φr
            STATIC_RS1,
            STATIC_RS2,
            -30,
            0,
            True
        )

        # Build initial 3D radiation surface
        init_vec = build_vector_figure_visible(init_data)

        # Build initial phase mask
        init_ph = build_phase_figure(init_data)

        # Build initial beam projection
        init_be = build_beam_figure(init_data, -30, 0)

        print("Initial figures created successfully.")

    except Exception:

        print("Initial figure generation failed. Using placeholders.")
        traceback.print_exc()

        # Fallback 3D placeholder
        init_vec = go.Figure()
        init_vec.update_layout(scene=dict(aspectmode='cube'))

        # Empty placeholders
        init_ph = go.Figure()
        init_be = go.Figure()

    # ================= BUILD DASH LAYOUT =================
    dash_app.layout = html.Div([

        # Application title
        html.H3("Far-field / Reflectarray Visualizer (Dash)"),

        # Main horizontal layout container
        html.Div([

            # ================= LEFT CONTROL PANEL =================
            html.Div([

                # θi control label
                html.Label("θi (deg)"),

                # θi slider
                dcc.Slider(
                    id='theta-inc-slider',
                    min=0, max=90,
                    step=1,
                    value=30,
                    marks={0:'0',30:'30',60:'60',90:'90'}
                ),

                # θi numeric input
                dcc.Input(
                    id='theta-inc-input',
                    type='number',
                    min=0, max=90,
                    step=1,
                    value=30
                ),

                html.Br(),

                # φi controls
                html.Label("φi (deg)"),
                dcc.Slider(
                    id='phi-inc-slider',
                    min=0, max=180,
                    step=1,
                    value=90,
                    marks={0:'0',90:'90',180:'180'}
                ),
                dcc.Input(
                    id='phi-inc-input',
                    type='number',
                    min=0, max=180,
                    step=1,
                    value=90
                ),

                html.Br(),

                # θr controls
                html.Label("θr (deg)"),
                dcc.Slider(
                    id='theta-ref-slider',
                    min=0, max=90,
                    step=1,
                    value=90,
                    marks={0:'0',30:'30',60:'60',90:'90'}
                ),
                dcc.Input(
                    id='theta-ref-input',
                    type='number',
                    min=0, max=90,
                    step=1,
                    value=90
                ),

                html.Br(),

                # φr controls
                html.Label("φr (deg)"),
                dcc.Slider(
                    id='phi-ref-slider',
                    min=0, max=180,
                    step=1,
                    value=150,
                    marks={0:'0',90:'90',180:'180'}
                ),
                dcc.Input(
                    id='phi-ref-input',
                    type='number',
                    min=0, max=180,
                    step=1,
                    value=150
                ),

                html.Br(),

                # Display fixed RIS size
                html.P(
                    f"RIS size: {STATIC_RS1} × {STATIC_RS2} elements (fixed)",
                    style={'fontWeight':'bold'}
                ),

                # Dynamic range controls
                dr_control('dr1', 'dr2', dr_min_default=-30, dr_max_default=0),

                # Randomization toggle
                randomization_control('rand', default='On'),

                html.Br(),

                # Upload grid image (optional)
                dcc.Upload(
                    id='upload-image',
                    children=html.Button('Upload grid.png (optional)')
                ),

                html.Br(),

                # Run computation button
                html.Button(
                    'Run',
                    id='run-button',
                    n_clicks=0,
                    style={'backgroundColor':'#f5da81'}
                ),

                # Status message display
                html.Div(
                    id='status',
                    style={'marginTop':'6px','color':'green'}
                )

            ], style={
                'width':'30%',
                'display':'inline-block',
                'verticalAlign':'top',
                'padding':'10px',
                'boxSizing':'border-box'
            }),

            # ================= RIGHT GRAPH PANEL =================
            html.Div([

                # 3D radiation plot
                dcc.Graph(
                    id='vector-graph',
                    figure=init_vec,
                    config={'displayModeBar': False},
                    style={'height':'460px', 'width':'100%'}
                ),

                # Phase mask plot
                dcc.Graph(
                    id='phase-graph',
                    figure=init_ph,
                    style={'height':'360px', 'width':'100%'}
                ),

                # Beam projection plot
                dcc.Graph(
                    id='beam-graph',
                    figure=init_be,
                    style={'height':'420px', 'width':'100%'}
                ),

            ], style={
                'width':'70%',
                'display':'inline-block',
                'padding':'0 12px',
                'boxSizing':'border-box'
            }),
        ])
    ], style={'fontFamily':'Arial, sans-serif'})

    print("Dash layout constructed successfully.")