# app/dash_apps/nearfield.py
# Main Dash application for Near-field Reflectarray Visualizer

# Import Dash core classes and callback utilities
from dash import Dash, html, dcc, Input, Output, State

# Import dash namespace (for dash.no_update)
import dash

# Import Plotly graph objects for building figures
import plotly.graph_objects as go

# Import traceback for detailed error printing
import traceback


# Utility function to create empty placeholder figure
def empty_figure(title="", xaxis_title=None, yaxis_title=None, template="plotly_white"):

    # Print debug info
    print("Creating empty figure with title:", title)

    # Create empty Plotly figure
    fig = go.Figure()

    # Set title
    fig.update_layout(title=title)

    # Optionally set axis titles
    if xaxis_title or yaxis_title:
        fig.update_layout(
            xaxis_title=xaxis_title or "",
            yaxis_title=yaxis_title or ""
        )

    # Apply visual theme
    fig.update_layout(template=template)

    # Return figure
    return fig


# Build dynamic range control inputs
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

# Import backend nearfield computation functions and plotting helpers
from app.services.nearfield_core import (
    compute_nearfield,             # Core near-field EM computation
    STATIC_RS1,                    # Fixed RIS X dimension
    STATIC_RS2,                    # Fixed RIS Y dimension
    build_vector_figure,           # 3D vector visualization
    build_phase_figure,            # Phase mask visualization
    build_illumination_figure,     # Illumination map visualization
    build_xy_figure,               # XY cut visualization
    build_yz_figure                # YZ cut visualization
)


# Create Dash application for Near-field visualization
def create_nearfield_dash(server, url_base_pathname="/dash/near_field/"):

    print("\n========== INITIALIZING NEARFIELD DASH ==========")
    print("URL base path:", url_base_pathname)

    # Create Dash app instance
    dash_app = Dash(
        __name__,
        server=server,
        url_base_pathname=url_base_pathname,
        suppress_callback_exceptions=True
    )

    # ---------------- HELPER: FORCE 2D FIGURE SQUARE ----------------
    def _enforce_square_2d(fig, size_px=520):

        """
        Force a 2D Plotly figure to render square (equal x/y scale and fixed pixel size).
        """

        print("Applying square enforcement (2D). Size:", size_px)

        try:
            # Force equal axis scaling
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

            # Fix pixel size and margins
            fig.update_layout(
                autosize=False,
                width=size_px,
                height=size_px,
                margin=dict(l=30, r=10, t=40, b=30)
            )

        except Exception:
            # Ignore if figure not compatible
            pass

        return fig


    # ---------------- HELPER: FORCE 3D FIGURE SQUARE ----------------
    def _enforce_square_3d(fig, size_px=520):

        """Ensure 3D scene uses equal aspect (cube) and set layout size."""

        print("Applying square enforcement (3D). Size:", size_px)

        try:
            # Set 3D aspect mode to cube
            fig.update_layout(scene=dict(aspectmode='cube'))

            # Fix pixel size and margins
            fig.update_layout(
                autosize=False,
                width=size_px,
                height=size_px,
                margin=dict(l=30, r=10, t=40, b=30)
            )

        except Exception:
            pass

        return fig


    # ---------------- INITIAL SAFE FIGURES ----------------
    try:

        print("Computing initial nearfield figures...")

        # Run initial nearfield computation with default values
        data0 = compute_nearfield(
            0,    # xi
            145,  # yi
            250,  # zi
            0,    # xr
            0,    # yr
            1300, # zr
            0,    # zcut
            dr1=-20,
            dr2=0,
            randomize=True
        )

        # Build initial vector figure
        init_vec = build_vector_figure(data0)
        init_vec = _enforce_square_3d(init_vec, size_px=480)

        # Build initial phase mask
        init_phase = build_phase_figure(data0)
        init_phase = _enforce_square_2d(init_phase, size_px=480)

        # Build initial illumination plot
        init_illum = build_illumination_figure(data0)
        init_illum = _enforce_square_2d(init_illum, size_px=480)

        # Build initial XY cut
        init_xy = build_xy_figure(data0)
        init_xy = _enforce_square_2d(init_xy, size_px=480)

        # Build initial YZ cut
        init_yz = build_yz_figure(data0)
        init_yz = _enforce_square_2d(init_yz, size_px=480)

        print("Initial figures generated successfully.")

    except Exception:

        print("Initial figure generation failed. Using placeholders.")
        traceback.print_exc()

        # Create fallback empty figures
        init_vec = empty_figure()
        init_vec.update_layout(scene=dict(aspectmode='cube'))
        init_vec = _enforce_square_3d(init_vec, size_px=480)

        init_phase = _enforce_square_2d(empty_figure(), size_px=480)
        init_illum = _enforce_square_2d(empty_figure(), size_px=480)
        init_xy = _enforce_square_2d(empty_figure(), size_px=480)
        init_yz = _enforce_square_2d(empty_figure(), size_px=480)

    # ---------------- DASH LAYOUT ----------------
    dash_app.layout = html.Div([

        # Page title
        html.H3("Near-field / Reflectarray Visualizer (Dash)"),

        # Main horizontal layout container
        html.Div([

            # ================= LEFT CONTROL PANEL =================
            html.Div([

                # xi input
                html.Label("xᵢ (mm)"),
                dcc.Input(id='xi', type='number', value=0),

                html.Br(), html.Br(),

                # yi input
                html.Label("yᵢ (mm)"),
                dcc.Input(id='yi', type='number', value=145),

                html.Br(), html.Br(),

                # zi input
                html.Label("zᵢ (mm)"),
                dcc.Input(id='zi', type='number', value=250),

                html.Br(), html.Br(),

                # xr input
                html.Label("xᵣ (mm)"),
                dcc.Input(id='xr', type='number', value=0),

                html.Br(), html.Br(),

                # yr input
                html.Label("yᵣ (mm)"),
                dcc.Input(id='yr', type='number', value=0),

                html.Br(), html.Br(),

                # zr input
                html.Label("zᵣ (mm)"),
                dcc.Input(id='zr', type='number', value=1300),

                html.Br(), html.Br(),

                # Z-cut input
                html.Label("Z-cut (for plotting)"),
                dcc.Input(id='zcut', type='number', value=0),

                html.Br(), html.Br(),

                # Randomization toggle
                randomization_control('rand', default='On'),

                html.Br(), html.Br(),

                # Dynamic range controls
                dr_control('dr1', 'dr2', dr_min_default=-20, dr_max_default=0),

                html.Br(), html.Br(),

                # Computed angles display (read-only)
                html.Div([

                    html.Label("θi (deg)"),
                    dcc.Input(id='theta_i', type='number', readOnly=True),

                    html.Br(), html.Br(),

                    html.Label("φi (deg)"),
                    dcc.Input(id='phi_i', type='number', readOnly=True),

                    html.Br(), html.Br(),

                    html.Label("θr (deg)"),
                    dcc.Input(id='theta_r', type='number', readOnly=True),

                    html.Br(), html.Br(),

                    html.Label("φr (deg)"),
                    dcc.Input(id='phi_r', type='number', readOnly=True),

                ]),

                html.Br(), html.Br(),

                # Update button
                html.Button(
                    'Update',
                    id='update-button',
                    n_clicks=0,
                    style={'backgroundColor':'#f5da81'}
                ),

                # Status message
                html.Div(
                    id='status',
                    style={'marginTop':'6px','color':'green'}
                ),

            ], style={
                'width':'30%',
                'display':'inline-block',
                'verticalAlign':'top',
                'padding':'10px',
                'boxSizing':'border-box'
            }),

            # ================= RIGHT GRAPH PANEL =================
            html.Div([

                # 3D vector plot
                dcc.Graph(
                    id='vector-graph',
                    figure=init_vec,
                    config={'displayModeBar': False},
                    style={
                        'height': '480px',
                        'width': '100%',
                        'marginBottom': '40px'
                    }
                ),

                # Phase mask plot
                dcc.Graph(
                    id='phase-graph',
                    figure=init_phase,
                    style={
                        'height': '480px',
                        'width': '100%',
                        'marginBottom': '40px'
                    }
                ),

                # Illumination plot
                dcc.Graph(
                    id='illum-graph',
                    figure=init_illum,
                    style={
                        'height': '480px',
                        'width': '100%',
                        'marginBottom': '40px'
                    }
                ),

                # XY cut plot
                dcc.Graph(
                    id='xy-graph',
                    figure=init_xy,
                    style={
                        'height': '480px',
                        'width': '100%',
                        'marginBottom': '40px'
                    }
                ),

                # YZ cut plot
                dcc.Graph(
                    id='yz-graph',
                    figure=init_yz,
                    style={
                        'height': '480px',
                        'width': '100%'
                    }
                ),

            ], style={
                'width': '70%',
                'display': 'inline-block',
                'padding': '0 12px',
                'boxSizing': 'border-box',
                'verticalAlign': 'top'
            })

        ])

    ])

    print("Nearfield Dash layout constructed successfully.")


    # ================= UPDATE CALLBACK =================
    @dash_app.callback(
        [
            # Updated 3D vector figure
            Output('vector-graph', 'figure'),

            # Updated phase mask figure
            Output('phase-graph', 'figure'),

            # Updated illumination figure
            Output('illum-graph', 'figure'),

            # Updated XY cut figure
            Output('xy-graph', 'figure'),

            # Updated YZ cut figure
            Output('yz-graph', 'figure'),

            # Status message
            Output('status', 'children'),

            # Computed θi
            Output('theta_i', 'value'),

            # Computed φi
            Output('phi_i', 'value'),

            # Computed θr
            Output('theta_r', 'value'),

            # Computed φr
            Output('phi_r', 'value')
        ],
        [
            # Trigger: Update button
            Input('update-button', 'n_clicks')
        ],
        [
            # State inputs
            State('xi','value'),
            State('yi','value'),
            State('zi','value'),
            State('xr','value'),
            State('yr','value'),
            State('zr','value'),
            State('zcut','value'),
            State('dr1','value'),
            State('dr2','value'),
            State('rand','value')
        ]
    )

    # Callback function definition
    def on_update(n_clicks,
                  xi, yi, zi,
                  xr, yr, zr,
                  zcut,
                  dr1, dr2,
                  rand):

        print("\n========== NEARFIELD DASH UPDATE ==========")
        print("Update button clicked:", n_clicks)

        try:

            # Determine randomization flag
            randomize = (rand == 'On')

            print("Randomization:", randomize)
            print("Tx:", (xi, yi, zi))
            print("Rx:", (xr, yr, zr))
            print("Z-cut:", zcut)
            print("Dynamic range:", dr1, dr2)

            # Call backend nearfield computation
            data = compute_nearfield(
                xi, yi, zi,
                xr, yr, zr,
                zcut,
                dr1=float(dr1),
                dr2=float(dr2),
                randomize=randomize
            )

            # If computation failed
            if data is None or "error" in data:

                msg = data.get("error", "compute error") if isinstance(data, dict) else "compute error"

                print("Backend error:", msg)

                return (
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    msg,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update
                )

            # Build updated figures
            fig_vec = build_vector_figure(data)
            fig_vec = _enforce_square_3d(fig_vec, size_px=480)

            fig_phase = build_phase_figure(data)
            fig_phase = _enforce_square_2d(fig_phase, size_px=480)

            fig_illum = build_illumination_figure(data)
            fig_illum = _enforce_square_2d(fig_illum, size_px=480)

            fig_xy = build_xy_figure(data)
            fig_xy = _enforce_square_2d(fig_xy, size_px=480)

            fig_yz = build_yz_figure(data)
            fig_yz = _enforce_square_2d(fig_yz, size_px=480)

            # Extract computed angle parameters
            params = data.get("params", {})

            # Status message
            status = f"Computed nearfield — RIS={STATIC_RS1}×{STATIC_RS2}"

            print("Computation successful.")
            print("========== NEARFIELD DASH UPDATE END ==========\n")

            # Return updated outputs
            return (
                fig_vec,
                fig_phase,
                fig_illum,
                fig_xy,
                fig_yz,
                status,
                params.get("theta_inc_deg"),
                params.get("phi_inc_deg"),
                params.get("theta_ref_deg"),
                params.get("phi_ref_deg")
            )

        except Exception as e:

            # Capture traceback
            tb = traceback.format_exc()

            print("Dash nearfield update error:", e)
            print(tb)

            # Create placeholder figure
            placeholder = _enforce_square_2d(empty_figure(), size_px=480)

            # Return placeholders and error message
            return (
                placeholder,
                placeholder,
                placeholder,
                placeholder,
                placeholder,
                f"Error: {str(e)}",
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update
            )

    print("========== NEARFIELD DASH READY ==========\n")

    # Return Dash app instance
    return dash_app