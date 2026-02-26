# app/dash_apps/farfield_range_of_beams_sub6.py
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import os, traceback, time

from app.services.farfieldrangeofbeamssub6_core import compute_farfield_codebook
from app.dash_apps.callback_helpers import empty_figure

def create_farfield_range_of_beams_sub6_dash(server=None, url_base_pathname="/dash/far_field_range_of_beams_sub6/"):
    dash_app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP],
                    url_base_pathname=url_base_pathname, suppress_callback_exceptions=True)
    dash_app.title = "MetaMaDe — Far Field Range of Beams (Sub-6 RIS)"

    dash_app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("MetaMaDe — Far Field Range of Beams (Sub-6 RIS)"),
                html.Hr(),

                dbc.Label("Incident θ (deg)"),
                dcc.Input(id='ti', type='number', value=0, step=1),html.Br(),html.Br(),

                dbc.Label("Incident φ (deg)", className="mt-2"),
                dcc.Input(id='pi', type='number', value=0, step=1),

                html.Hr(),
                dbc.Label("Reflection θ"),html.Br(),html.Br(),
                dbc.Label("Start"),
                dcc.Input(id='trstart', type='number', value=0, step=1),html.Br(),html.Br(),
                dbc.Label("Step"),
                dcc.Input(id='trsteps', type='number', value=1, step=1),html.Br(),html.Br(),
                dbc.Label("Stop"),
                dcc.Input(id='trstop', type='number', value=10, step=1),

                html.Hr(),
                dbc.Label("Reflection φ:"),html.Br(),html.Br(),
                dbc.Label("Start"),
                dcc.Input(id='prstart', type='number', value=0, step=1),html.Br(),html.Br(),
                dbc.Label("Step"),
                dcc.Input(id='prsteps', type='number', value=1, step=1),html.Br(),html.Br(),
                dbc.Label("Stop"),
                dcc.Input(id='prstop', type='number', value=10, step=1),

                html.Hr(),
                dbc.Label("RIS Size (RS1 x RS2)"),html.Br(),html.Br(),
                dcc.Input(id='RS1', type='number', value=28, step=1),
                dcc.Input(id='RS2', type='number', value=36, step=1),

                html.Hr(),
                dbc.Label("Dynamic Range (dB)"),html.Br(),html.Br(),
                dcc.Input(id='DR1', type='number', value=-30, step=1),
                dcc.Input(id='DR2', type='number', value=0, step=1),

                html.Hr(),
                dbc.Checklist(
                    options=[{"label": "Fast mode (coarser grid for debugging)", "value": "fast"}],
                    value=[],
                    id="dev-mode",
                    inline=True,
                ),

                html.Div(className="mt-3"),
                dbc.Button("Update", id='update-btn', color='primary', n_clicks=0),
                dbc.Button("Clear Plot", id='clear-btn', color='secondary', n_clicks=0, className="ms-2"),

                html.Div(className="mt-2"),
                dcc.Download(id="download-codebook"),
                html.Div(id='run-meta', className='mt-2 text-muted'),
            ], width=4),

            dbc.Col([
                dcc.Loading(
                    id="loading-plot",
                    type="circle",
                    children=dcc.Graph(id='uv-plot', figure=go.Figure())
                ),
                html.Div(className="mt-2"),
                dbc.Button("Download Codebook TXT", id='download-btn', color='success', className="mt-2"),
            ], width=8)
        ], className="mt-3")
    ], fluid=True)

    @dash_app.callback(
        Output('uv-plot', 'figure'),
        Output('run-meta', 'children'),
        Input('update-btn', 'n_clicks'),
        State('prstart', 'value'), State('prstop', 'value'), State('prsteps', 'value'),
        State('trstart', 'value'), State('trstop', 'value'), State('trsteps', 'value'),
        State('pi', 'value'), State('ti', 'value'),
        State('RS1', 'value'), State('RS2', 'value'),
        State('DR1', 'value'), State('DR2', 'value'),
        State('dev-mode', 'value'),
        prevent_initial_call=True
    )
    def on_update(n_clicks, prstart, prstop, prsteps, trstart, trstop, trsteps, pi, ti, RS1, RS2, DR1, DR2, dev_mode_val):
        import time, traceback
        start_time = time.time()
        print(">>> on_update called, n_clicks=", n_clicks)

        # Safe conversions
        try:
            RS1 = int(RS1) if RS1 is not None else 28
            RS2 = int(RS2) if RS2 is not None else 36
            prstart = float(prstart or 0.0)
            prstop  = float(prstop  or 10.0)
            prsteps = float(prsteps or 1.0)
            trstart = float(trstart or 0.0)
            trstop  = float(trstop  or 10.0)
            trsteps = float(trsteps or 1.0)
            pi = float(pi or 0.0)
            ti = float(ti or 0.0)
            DR1 = float(DR1 or -30.0)
            DR2 = float(DR2 or 0.0)
        except Exception as e:
            print("Parameter conversion error:", e)
            traceback.print_exc()
            fig_err = go.Figure()
            fig_err.add_annotation(text="Parameter conversion error — check server logs", showarrow=False)
            return fig_err, f"Parameter conversion error: {e}"

        # dev-mode handling: checklist returns list (or None)
        dev_mode = False
        try:
            dev_mode = bool(dev_mode_val and "fast" in dev_mode_val)
        except Exception:
            dev_mode = False

        params = {
            'prstart': prstart, 'prstop': prstop, 'prsteps': prsteps,
            'trstart': trstart, 'trstop': trstop, 'trsteps': trsteps,
            'pi': pi, 'ti': ti, 'RS1': RS1, 'RS2': RS2,
            'DR1': DR1, 'DR2': DR2, 'dev_mode': dev_mode
        }

        print("Calling core with params:", {k: params[k] for k in ['RS1','RS2','dev_mode','prstart','prstop','trstart','trstop']})
        try:
            result = compute_farfield_codebook(params)
            if result is None:
                raise RuntimeError("compute_farfield_codebook returned None")
            fig = result.get('figure', None)
            meta = result.get('meta', {})
            elapsed = time.time() - start_time

            # if fig is None or has no traces, build a fallback figure so UI is never blank
            if fig is None or (hasattr(fig, 'data') and len(fig.data) == 0):
                print("Core returned empty figure — building fallback visible figure")
                # simple synthetic visible fallback
                uu = np.linspace(-1, 1, 101)
                vv = np.linspace(-1, 1, 101)
                Uu, Vv = np.meshgrid(uu, vv)
                R = np.sqrt(Uu**2 + Vv**2)
                Z = -40 * np.ones_like(Uu)
                inside = R <= 1.0
                Z[inside] = -10 + 8.0 * np.exp(-8.0 * (Uu[inside]**2 + Vv[inside]**2))
                fig_fallback = empty_figure(title="Beam pattern (fallback)")
                fig_fallback.add_trace(go.Heatmap(z=Z, x=uu, y=vv, zmin=DR1, zmax=DR2, colorscale='Viridis',
                                                  colorbar=dict(title="Magnitude dB")))
                fig_fallback.update_layout(xaxis=dict(range=[-1,1]), yaxis=dict(range=[-1,1], scaleanchor="x", scaleratio=1))
                meta_text = f"(fallback) core returned empty — took {elapsed:.1f}s"
                print("Returning fallback figure")
                return fig_fallback, meta_text

            meta_text = f"Generated {meta.get('masks_generated',0)} masks — codewords: {meta.get('codewords',0)} (took {elapsed:.1f}s)"
            print("compute_farfield_codebook succeeded:", meta_text)
            return fig, meta_text

        except Exception as e:
            print("Error in compute_farfield_codebook:", e)
            traceback.print_exc()
            # visible fallback
            uu = np.linspace(-1, 1, 101)
            vv = np.linspace(-1, 1, 101)
            Uu, Vv = np.meshgrid(uu, vv)
            R = np.sqrt(Uu**2 + Vv**2)
            Z = -40 * np.ones_like(Uu)
            inside = R <= 1.0
            Z[inside] = -10 + 8.0 * np.exp(-8.0 * (Uu[inside]**2 + Vv[inside]**2))
            fig_err = empty_figure(title="Beam pattern (error fallback)")
            fig_err.add_trace(go.Heatmap(z=Z, x=uu, y=vv, zmin=DR1, zmax=DR2, colorscale='Viridis',
                                         colorbar=dict(title="Magnitude dB")))
            fig_err.update_layout(xaxis=dict(range=[-1,1]), yaxis=dict(range=[-1,1], scaleanchor="x", scaleratio=1))
            return fig_err, f"Error: {str(e)} — see server logs"

    @dash_app.callback(
        Output("download-codebook", "data"),
        Input("download-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def download_codebook(n_clicks):
        filename = "Far_Field_Codebook_Sub6.txt"
        if not os.path.exists(filename):
            return dcc.send_string("No codebook found. Please run Update first.", filename="error.txt")
        with open(filename, 'r') as f:
            s = f.read()
        return dcc.send_string(s, filename)

    @dash_app.callback(
        Output('uv-plot', 'figure'),
        Output('run-meta', 'children'),
        Input('clear-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def on_clear(n_clicks):
        fig = go.Figure()
        return fig, "Plot cleared."

    return dash_app

