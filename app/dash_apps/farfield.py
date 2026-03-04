# app/dash_apps/farfield.py
import traceback
from dash import Dash, html, dcc, Input, Output, State
import dash
import plotly.graph_objects as go


def register_sync_pair(app, slider_id, input_id, minv, maxv, cast=int):
    """Register a sync callback between a slider and numeric input.

    Ensures the two controls remain in sync and clamps input to [minv, maxv].
    """
    @app.callback(
        Output(slider_id, "value"),
        Output(input_id, "value"),
        Input(slider_id, "value"),
        Input(input_id, "value"),
    )
    def _sync(slider_val, input_val):
        triggered_id = dash.callback_context.triggered_id
        if triggered_id == slider_id:
            return slider_val, slider_val
        if triggered_id == input_id:
            if input_val is None:
                return dash.no_update, dash.no_update
            try:
                v = cast(input_val)
            except Exception:
                return dash.no_update, dash.no_update
            v = max(minv, min(maxv, v))
            return v, v
        return dash.no_update, dash.no_update


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


def randomization_control(rand_id: str, default: str = "On"):
    """Return a small radio control for randomization On/Off."""
    return html.Div([
        html.Label("Randomization"),
        dcc.RadioItems(id=rand_id, options=[{"label": "On", "value": "On"}, {"label": "Off", "value": "Off"}], value=default, inline=True),
    ], style={"marginTop": "6px"})
from app.services.farfield_core import (
    compute_fields,
    build_vector_figure_visible,
    build_phase_figure,
    build_beam_figure,
    STATIC_RS1,
    STATIC_RS2
)

def create_farfield_dash(server, url_base_pathname="/dash/far_field/"):
    dash_app = Dash(__name__, server=server, url_base_pathname=url_base_pathname,
                    suppress_callback_exceptions=True)

    # initial defaults & build initial figures safely
    try:
        init_data = compute_fields(30, 90, 90, 150, STATIC_RS1, STATIC_RS2, -30, 0, True)
        init_vec = build_vector_figure_visible(init_data)
        init_ph = build_phase_figure(init_data)
        init_be = build_beam_figure(init_data, -30, 0)
    except Exception:
        init_vec = go.Figure(); init_vec.update_layout(scene=dict(aspectmode='cube'))
        init_ph = go.Figure()
        init_be = go.Figure()

    dash_app.layout = html.Div([
        html.H3("Far-field / Reflectarray Visualizer (Dash)"),
        html.Div([
            html.Div([  # controls
                html.Label("θi (deg)"),
                dcc.Slider(id='theta-inc-slider', min=0, max=90, step=1, value=30,
                           marks={0:'0',30:'30',60:'60',90:'90'}),
                dcc.Input(id='theta-inc-input', type='number', min=0, max=90, step=1, value=30),
                html.Br(),
                html.Label("φi (deg)"),
                dcc.Slider(id='phi-inc-slider', min=0, max=180, step=1, value=90,
                           marks={0:'0',90:'90',180:'180'}),
                dcc.Input(id='phi-inc-input', type='number', min=0, max=180, step=1, value=90),
                html.Br(),
                html.Label("θr (deg)"),
                dcc.Slider(id='theta-ref-slider', min=0, max=90, step=1, value=90,
                           marks={0:'0',30:'30',60:'60',90:'90'}),
                dcc.Input(id='theta-ref-input', type='number', min=0, max=90, step=1, value=90),
                html.Br(),
                html.Label("φr (deg)"),
                dcc.Slider(id='phi-ref-slider', min=0, max=180, step=1, value=150,
                           marks={0:'0',90:'90',180:'180'}),
                dcc.Input(id='phi-ref-input', type='number', min=0, max=180, step=1, value=150),
                html.Br(),
                html.P(f"RIS size: {STATIC_RS1} × {STATIC_RS2} elements (fixed)", style={'fontWeight':'bold'}),
                dr_control('dr1', 'dr2', dr_min_default=-30, dr_max_default=0),
                randomization_control('rand', default='On'),
                html.Br(),
                dcc.Upload(id='upload-image', children=html.Button('Upload grid.png (optional)')),
                html.Br(),
                html.Button('Run', id='run-button', n_clicks=0, style={'backgroundColor':'#f5da81'}),
                html.Div(id='status', style={'marginTop':'6px','color':'green'})
            ], style={'width':'30%','display':'inline-block','verticalAlign':'top','padding':'10px','boxSizing':'border-box'}),

            html.Div([  # graphs
                dcc.Graph(id='vector-graph', figure=init_vec, config={'displayModeBar': False},
                          style={'height':'460px', 'width':'100%'}),
                dcc.Graph(id='phase-graph', figure=init_ph, style={'height':'360px', 'width':'100%'}),
                dcc.Graph(id='beam-graph', figure=init_be, style={'height':'420px', 'width':'100%'}),
            ], style={'width':'70%','display':'inline-block','padding':'0 12px','boxSizing':'border-box'}),
        ])
    ], style={'fontFamily':'Arial, sans-serif'})

    # register sync callbacks using helper to avoid repetition
    register_sync_pair(dash_app, 'theta-inc-slider', 'theta-inc-input', 0, 90, int)
    register_sync_pair(dash_app, 'phi-inc-slider', 'phi-inc-input', 0, 180, int)
    register_sync_pair(dash_app, 'theta-ref-slider', 'theta-ref-input', 0, 90, int)
    register_sync_pair(dash_app, 'phi-ref-slider', 'phi-ref-input', 0, 180, int)

    # Run callback
    @dash_app.callback(
        [Output('vector-graph', 'figure'),
         Output('phase-graph', 'figure'),
         Output('beam-graph', 'figure'),
         Output('status', 'children')],
        [Input('run-button', 'n_clicks')],
        [State('theta-inc-slider','value'),
         State('phi-inc-slider','value'),
         State('theta-ref-slider','value'),
         State('phi-ref-slider','value'),
         State('dr1','value'),
         State('dr2','value'),
         State('rand','value'),
         State('upload-image','contents')]
    )
    def on_run(n_clicks, theta_inc, phi_inc, theta_ref, phi_ref, dr1, dr2, rand, uploaded_contents):
        if dr1 is None or dr2 is None:
            return dash.no_update, dash.no_update, dash.no_update, "Set DR1/DR2"
        try:
            randomize = (rand == 'On')
            DR1_f = float(dr1); DR2_f = float(dr2)
            # delegate compute_fields
            data = compute_fields(theta_inc, phi_inc, theta_ref, phi_ref, STATIC_RS1, STATIC_RS2, DR1_f, DR2_f, randomize)
            if data is None or "error" in data:
                status = data.get("error", "Compute error")
                return dash.no_update, dash.no_update, dash.no_update, status

            fig_vec = build_vector_figure_visible(data)
            fig_ph = build_phase_figure(data)
            fig_beam = build_beam_figure(data, DR1_f, DR2_f)
            status = f"Computed: θi={theta_inc}°, φi={phi_inc}°, θr={theta_ref}°, φr={phi_ref}° — RS=({STATIC_RS1}×{STATIC_RS2}) [fixed]"
            return fig_vec, fig_ph, fig_beam, status

        except Exception as err:
            print("Run callback error:", err); traceback.print_exc()
            placeholder = empty_figure()
            placeholder.update_layout(scene=dict(aspectmode='cube'))
            return placeholder, placeholder, placeholder, f"Error: {str(err)}"

    return dash_app
