from dash import Output, Input
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
