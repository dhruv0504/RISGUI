# app/services/farfield_core.py
# Far-Field compute & plotting service (wrapped from your provided script)

import io
import sys
import traceback
import numpy as np
import plotly.graph_objects as go

# Small helpers
def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

# sample random vector from your paste
phi_ele_rand_sample = np.array([
 0.219827130698740, 2.98461967174048, 0.497030697989493, 0.899896229536556,
 2.15867977892526, 0.443435248995979, 1.60876765916016, 2.26611537562385,
 2.91805305438595, 2.29997207687589, 2.35571703325974, 1.27963227277936,
 0.752385351454657, 1.63630002613963, 0.688251825085977, 2.64643965036121
])

# Default static RIS
STATIC_RS1 = 32
STATIC_RS2 = 32

def compute_fields(theta_inc_deg, phi_inc_deg,
                   theta_ref_deg, phi_ref_deg,
                   RS1_arg, RS2_arg,
                   DR1, DR2, randomize,
                   fc=28.5, ant_size_x=5.25, ant_size_y=5.25):
    """
    Core compute function moved from your script.
    Returns dict with masks, element coords, r/r_clip, and angular grids.
    """
    try:
        # Protect fc from zero / falsy values
        if not fc:
            fc = 28.5

        # Force static RIS size (override passed RS1_arg/RS2_arg)
        RS1 = 32  # columns (x)
        RS2 = 32  # rows    (y)

        # Convert angles to radians (consistent usage)
        ti = np.deg2rad(theta_inc_deg)
        pi = np.deg2rad(phi_inc_deg)
        tr = np.deg2rad(theta_ref_deg)
        pr = np.deg2rad(phi_ref_deg)

        # Incident / reflected unit vectors
        incident_vector = np.array([np.sin(ti) * np.cos(pi),
                                    np.sin(ti) * np.sin(pi),
                                    np.cos(ti)])
        reflected_vector = np.array([np.sin(tr) * np.cos(pr),
                                     np.sin(tr) * np.sin(pr),
                                     np.cos(tr)])

        # Wavenumber (fc in MHz -> lambda in meters)
        lambda0 = 300.0 / fc
        k0 = 2.0 * np.pi / lambda0

        # Element coordinates (centered grid)
        R_x = ant_size_x * RS1
        R_y = ant_size_y * RS2
        x_vals = np.linspace(-R_x/2 + ant_size_x/2, R_x/2 - ant_size_x/2, RS1)
        y_vals = np.linspace(-R_y/2 + ant_size_y/2, R_y/2 - ant_size_y/2, RS2)
        X, Y = np.meshgrid(x_vals, y_vals)   # shape (RS2, RS1)

        # Feed phase term (kr) using feed direction (incident angles)
        kx_feed = k0 * np.sin(ti) * np.cos(pi)
        ky_feed = k0 * np.sin(ti) * np.sin(pi)
        kr = wrap_to_pi(kx_feed * X + ky_feed * Y)

        # Element randomization phase (per-element). Ensure correct shape (RS2,RS1).
        if not randomize:
            phi_ele_rand = np.zeros_like(X)
        else:
            nels = RS1 * RS2
            v = phi_ele_rand_sample.copy().ravel()
            if v.size < nels:
                v = np.tile(v, int(np.ceil(nels / v.size)))[:nels]
            else:
                v = v[:nels]
            phi_ele_rand = v.reshape((RS2, RS1))

        # Beam k-vector for target/reflected direction
        kx_beam = k0 * np.sin(tr) * np.cos(pr)
        ky_beam = k0 * np.sin(tr) * np.sin(pr)
        k_beam_r = wrap_to_pi(-(kx_beam * X + ky_beam * Y))

        # Compute required phase shift and 1-bit quantization (0 or pi)
        phase_shift = wrap_to_pi(k_beam_r - kr - phi_ele_rand)
        phase_shift_q = np.pi * (np.abs(phase_shift) >= (np.pi / 2))
        mask = phase_shift_q / np.pi   # 0 or 1

        # Observation grid (theta, phi)
        theta = np.arange(-90, 91, 1)   # -90..90 deg
        phi = np.arange(0, 361, 1)      # 0..360 deg
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)

        # Create observation direction cosines (phi outer, theta inner)
        PHI, THETA = np.meshgrid(phi_rad, theta_rad, indexing='ij')  # shape (len(phi), len(theta))
        U = np.sin(THETA) * np.cos(PHI)
        V = np.sin(THETA) * np.sin(PHI)

        # Static phase per element (base)
        static_phase = kr + phase_shift_q + phi_ele_rand

        # Vectorized field computation:
        Xv = X.ravel()    # (Nelem,)
        Yv = Y.ravel()
        base = static_phase.ravel()

        kX = k0 * Xv
        kY = k0 * Yv

        Uv = U.ravel()    # (Nobs,)
        Vv = V.ravel()

        # Build phase matrix (Nelem x Nobs). If memory is tight, compute in chunks.
        phase_mat = base[:, None] + (kX[:, None] * Uv[None, :]) + (kY[:, None] * Vv[None, :])

        # Sum element contributions for each observation direction
        E_vec = np.sum(np.exp(1j * phase_mat), axis=0)
        E_rad = E_vec.reshape(U.shape)   # (len(phi), len(theta))

        # dB scaling, normalization and clipping
        eps = 1e-12
        r = 20.0 * np.log10(np.abs(E_rad) + eps)
        r_norm = r - np.max(r)
        r_clip = np.clip(r_norm, DR1, DR2)

        return {
            "incident_vector": incident_vector,
            "reflected_vector": reflected_vector,
            "X": X, "Y": Y,
            "mask": mask,
            "r": r,
            "r_clip": r_clip,
            "x_vals": x_vals,
            "y_vals": y_vals,
            "theta": theta,
            "phi": phi
        }
    except Exception as e:
        # Return a minimal failure dictionary with an exception trace in status
        tb = traceback.format_exc()
        print("compute_fields error:", e)
        print(tb)
        return {"error": str(e), "trace": tb}

# Plot builders (moved & lightly adapted)

def build_vector_figure_visible(data, texture_rgb=None):
    iv = data["incident_vector"]
    rv = data["reflected_vector"]

    plane_half = 1.3
    ny, nx = data["X"].shape
    ny = max(ny, 8); nx = max(nx, 8)
    y_vals = np.linspace(-plane_half, plane_half, ny)
    z_vals = np.linspace(-plane_half, plane_half, nx)
    Yp, Zp = np.meshgrid(y_vals, z_vals)
    Xp = np.zeros_like(Yp)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=Xp, y=Yp, z=Zp,
        surfacecolor=np.ones_like(Xp),
        showscale=False, opacity=0.78,
        colorscale=[[0,'rgb(210,190,170)'], [1,'rgb(210,190,170)']],
        hoverinfo='skip', name='RIS plane'
    ))

    # grid lines
    for yv in y_vals:
        fig.add_trace(go.Scatter3d(x=[0,0], y=[yv,yv], z=[z_vals[0], z_vals[-1]],
                                   mode='lines', line=dict(color='saddlebrown', width=2), showlegend=False))
    for zv in z_vals:
        fig.add_trace(go.Scatter3d(x=[0,0], y=[y_vals[0], y_vals[-1]], z=[zv,zv],
                                   mode='lines', line=dict(color='saddlebrown', width=2), showlegend=False))

    def add_arrow_line(fig, start, vec, color, name=None):
        start = np.array(start, dtype=float)
        vec = np.array(vec, dtype=float)
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            vec = np.array([0.,0.,1.])
            norm = 1.0
        dir_u = vec / norm
        vis_len = 1.0
        tip = start + dir_u * vis_len
        line_end = start + dir_u * (vis_len * 0.86)
        fig.add_trace(go.Scatter3d(
            x=[start[0], line_end[0]],
            y=[start[1], line_end[1]],
            z=[start[2], line_end[2]],
            mode='lines', line=dict(color=color, width=6), name=name, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter3d(
            x=[tip[0]], y=[tip[1]], z=[tip[2]],
            mode='markers',
            marker=dict(size=10, color=color, symbol='diamond'),
            showlegend=False, hoverinfo='skip'
        ))

    add_arrow_line(fig, [1.5,0,0], -iv, 'red', name='Incident Vector')
    add_arrow_line(fig, [0,0,0], rv, 'blue', name='Reflected Vector')

    fig.update_layout(
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1.0, y=1.0, z=0.8),
            xaxis=dict(range=[-2,2], title='X', backgroundcolor='rgb(245,245,245)', gridcolor='lightgray'),
            yaxis=dict(range=[-2,2], title='Y', backgroundcolor='rgb(245,245,245)', gridcolor='lightgray'),
            zaxis=dict(range=[-2,2], title='Z', backgroundcolor='rgb(245,245,245)', gridcolor='lightgray'),
            camera=dict(eye=dict(x=1.3, y=-1.4, z=0.8))
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True
    )

    return fig

def build_phase_figure(data):
    mask = np.array(data["mask"])   # shape (RS2, RS1)
    x = data["x_vals"]
    y = data["y_vals"]

    mask_deg = mask * 180.0
    mask_deg = np.flipud(mask_deg)

    fig = go.Figure(
        go.Heatmap(
            z=mask_deg,
            x=x,
            y=y,
            colorscale=[
                [0.0, "#FFD700"], [0.499, "#FFD700"],
                [0.5, "#4B0082"], [1.0, "#4B0082"]
            ],
            zmin=0,
            zmax=180,
            showscale=False
        )
    )

    fig.update_layout(
        title="Randomized Phase Mask",
        xaxis_title="y-axis (wavelength)",
        yaxis_title="z-axis (wavelength)",
        yaxis_scaleanchor="x",
        margin=dict(l=60, r=20, t=40, b=50)
    )

    return fig

def build_beam_figure(data, DR1, DR2, grid_size=480):
    r = np.array(data["r_clip"])        # shape (len(phi), len(theta))
    theta = np.deg2rad(data["theta"])
    phi = np.deg2rad(data["phi"])

    TH, PH = np.meshgrid(theta, phi)   # TH,PH shape = (len(phi), len(theta))
    u = np.sin(TH) * np.cos(PH)
    v = np.sin(TH) * np.sin(PH)

    u_flat = u.ravel()
    v_flat = v.ravel()
    r_flat = r.ravel()

    inside = (u_flat**2 + v_flat**2) <= 1.0
    u_pts = u_flat[inside]
    v_pts = v_flat[inside]
    r_pts = r_flat[inside]

    xs = np.linspace(-1.0, 1.0, grid_size)   # v axis (horizontal)
    ys = np.linspace(-1.0, 1.0, grid_size)   # u axis (vertical)
    Z = np.full((grid_size, grid_size), np.nan, dtype=float)

    idx_x = np.clip(np.round(( (v_pts + 1.0) / 2.0 ) * (grid_size - 1)).astype(int), 0, grid_size-1)
    idx_y = np.clip(np.round(( (u_pts + 1.0) / 2.0 ) * (grid_size - 1)).astype(int), 0, grid_size-1)

    for xi, yi, rv in zip(idx_x, idx_y, r_pts):
        cur = Z[yi, xi]
        if np.isnan(cur) or (rv > cur):
            Z[yi, xi] = rv

    # fill small holes
    for _ in range(2):
        nan_mask = np.isnan(Z)
        if not nan_mask.any():
            break
        up = np.roll(Z, -1, axis=0)
        down = np.roll(Z, 1, axis=0)
        left = np.roll(Z, 1, axis=1)
        right = np.roll(Z, -1, axis=1)
        neighbor_sum = np.nansum(np.stack([up, down, left, right], axis=0), axis=0)
        neighbor_count = np.sum(~np.isnan(np.stack([up, down, left, right], axis=0)), axis=0)
        fill_vals = np.where(neighbor_count>0, neighbor_sum / neighbor_count, np.nan)
        Z = np.where(nan_mask & (~np.isnan(fill_vals)), fill_vals, Z)

    XV, YU = np.meshgrid(xs, ys)
    circle_mask = (XV**2 + YU**2) <= 1.0
    Z_masked = np.where(circle_mask, Z, np.nan)

    fig = go.Figure(
        go.Heatmap(
            x=xs,
            y=ys,
            z=Z_masked,
            colorscale="Viridis",
            zmin=DR1,
            zmax=DR2,
            colorbar=dict(title="Magnitude dB", ticks="outside"),
            hovertemplate="v=%{x:.3f}<br>u=%{y:.3f}<br>dB=%{z:.2f}<extra></extra>"
        )
    )

    ang = np.linspace(0, 2*np.pi, 512)
    circ_x = np.cos(ang)
    circ_y = np.sin(ang)
    fig.add_trace(
        go.Scatter(
            x=circ_x,
            y=circ_y,
            mode="lines",
            line=dict(color="black", width=1.8),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    fig.update_layout(
        title="Beam Pattern",
        xaxis=dict(title="v", range=[-1, 1], showgrid=False, zeroline=False),
        yaxis=dict(title="u", range=[-1, 1], scaleanchor="x", showgrid=False, zeroline=False),
        margin=dict(l=60, r=40, t=50, b=50),
        plot_bgcolor="white"
    )

    return fig
