# app/dash_apps/nearfield.py
from dash import Dash, html, dcc, Input, Output, State
import dash
import plotly.graph_objects as go
import traceback

from app.services.nearfield_core import (
    compute_nearfield,
    STATIC_RS1,
    STATIC_RS2,
    build_vector_figure,
    build_phase_figure,
    build_illumination_figure,
    build_xy_figure,
    build_yz_figure
)

def create_nearfield_dash(server, url_base_pathname="/dash/near_field/"):
    dash_app = Dash(__name__, server=server, url_base_pathname=url_base_pathname,
                    suppress_callback_exceptions=True)

    # initial safe figures
    try:
        data0 = compute_nearfield(0, 145, 250, 0, 0, 1300, 0, dr1=-20, dr2=0, randomize=True)
        init_vec = build_vector_figure(data0)
        init_phase = build_phase_figure(data0)
        init_illum = build_illumination_figure(data0)
        init_xy = build_xy_figure(data0)
        init_yz = build_yz_figure(data0)
    except Exception:
        init_vec = go.Figure(); init_vec.update_layout(scene=dict(aspectmode='cube'))
        init_phase = go.Figure()
        init_illum = go.Figure()
        init_xy = go.Figure()
        init_yz = go.Figure()

    dash_app.layout = html.Div([
        html.H3("Near-field / Reflectarray Visualizer (Dash)"),
        html.Div([
            html.Div([
                html.Label("xᵢ (mm)"), dcc.Input(id='xi', type='number', value=0),
                html.Br(),html.Br(),
                html.Label("yᵢ (mm)"), dcc.Input(id='yi', type='number', value=145),
                html.Br(),html.Br(),
                html.Label("zᵢ (mm)"), dcc.Input(id='zi', type='number', value=250),
                html.Br(),html.Br(),
                html.Label("xᵣ (mm)"), dcc.Input(id='xr', type='number', value=0),
                html.Br(),html.Br(),
                html.Label("yᵣ (mm)"), dcc.Input(id='yr', type='number', value=0),
                html.Br(),html.Br(),
                html.Label("zᵣ (mm)"), dcc.Input(id='zr', type='number', value=1300),
                html.Br(),html.Br(),
                html.Label("Z-cut (for plotting)"), dcc.Input(id='zcut', type='number', value=0),
                html.Br(),html.Br(),
                html.Label("Randomization"), dcc.RadioItems(id='rand', options=[{'label':'Off','value':'Off'},{'label':'On','value':'On'}], value='On'),
                html.Br(),html.Br(),
                html.Label("Dynamic Range"), dcc.Input(id='dr1', type='number', value=-20), html.Span(" to "), dcc.Input(id='dr2', type='number', value=0),
                html.Br(), html.Br(),
                html.Div([
                    html.Label("θi (deg)"), dcc.Input(id='theta_i', type='number', readOnly=True),
                    html.Br(),html.Br(),
                    html.Label("φi (deg)"), dcc.Input(id='phi_i', type='number', readOnly=True),
                    html.Br(),html.Br(),
                    html.Label("θr (deg)"), dcc.Input(id='theta_r', type='number', readOnly=True),
                    html.Br(),html.Br(),
                    html.Label("φr (deg)"), dcc.Input(id='phi_r', type='number', readOnly=True),
                ]),  # <-- MISSING COMMA FIXED HERE
                html.Br(),html.Br(),
                html.Button('Update', id='update-button', n_clicks=0, style={'backgroundColor':'#f5da81'}),
                html.Div(id='status', style={'marginTop':'6px','color':'green'}),
            ], style={'width':'30%','display':'inline-block','verticalAlign':'top','padding':'10px','boxSizing':'border-box'}),
            html.Div([
                dcc.Graph(id='vector-graph', figure=init_vec, config={'displayModeBar': False}, style={'height':'360px','width':'100%'}),
                dcc.Graph(id='phase-graph', figure=init_phase, style={'height':'280px','width':'100%'}),
                dcc.Graph(id='illum-graph', figure=init_illum, style={'height':'300px','width':'100%'}),
                dcc.Graph(id='xy-graph', figure=init_xy, style={'height':'360px','width':'100%'}),
                dcc.Graph(id='yz-graph', figure=init_yz, style={'height':'360px','width':'100%'}),
            ], style={'width':'70%','display':'inline-block','padding':'0 12px','boxSizing':'border-box'})
        ])
    ])

    @dash_app.callback(
        [Output('vector-graph', 'figure'),
         Output('phase-graph', 'figure'),
         Output('illum-graph', 'figure'),
         Output('xy-graph', 'figure'),
         Output('yz-graph', 'figure'),
         Output('status', 'children'),
         Output('theta_i', 'value'),
         Output('phi_i', 'value'),
         Output('theta_r', 'value'),
         Output('phi_r', 'value')],
        [Input('update-button', 'n_clicks')],
        [State('xi','value'),
         State('yi','value'),
         State('zi','value'),
         State('xr','value'),
         State('yr','value'),
         State('zr','value'),
         State('zcut','value'),
         State('dr1','value'),
         State('dr2','value'),
         State('rand','value')]
    )
    def on_update(n_clicks, xi, yi, zi, xr, yr, zr, zcut, dr1, dr2, rand):
        try:
            randomize = (rand == 'On')
            data = compute_nearfield(xi, yi, zi, xr, yr, zr, zcut,
                                     dr1=float(dr1), dr2=float(dr2),
                                     randomize=randomize)
            if data is None or "error" in data:
                msg = data.get("error", "compute error") if isinstance(data, dict) else "compute error"
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, msg, dash.no_update, dash.no_update, dash.no_update, dash.no_update

            fig_vec = build_vector_figure(data)
            fig_phase = build_phase_figure(data)
            fig_illum = build_illumination_figure(data)
            fig_xy = build_xy_figure(data)
            fig_yz = build_yz_figure(data)
            params = data.get("params", {})
            status = f"Computed nearfield — RIS={STATIC_RS1}×{STATIC_RS2}"
            return fig_vec, fig_phase, fig_illum, fig_xy, fig_yz, status, params.get("theta_inc_deg"), params.get("phi_inc_deg"), params.get("theta_ref_deg"), params.get("phi_ref_deg")
        except Exception as e:
            tb = traceback.format_exc()
            print("dash nearfield update error:", e)
            print(tb)
            placeholder = go.Figure()
            return placeholder, placeholder, placeholder, placeholder, placeholder, f"Error: {str(e)}", dash.no_update, dash.no_update, dash.no_update, dash.no_update

    return dash_app
