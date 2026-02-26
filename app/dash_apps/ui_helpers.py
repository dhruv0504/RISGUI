from dash import html, dcc


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
