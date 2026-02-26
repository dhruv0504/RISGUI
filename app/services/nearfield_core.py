# app/services/nearfield_core.py
"""
Near-field compute & plotting service based on your MATLAB implementation.
RIS size is fixed to 32x32. Exposes compute_nearfield(...) and helper builders.
"""

import traceback
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import os
from functools import lru_cache

# constants / default random sample (from your provided vector)
phi_ele_rand_sample = np.array([
 0.219827130698740, 2.98461967174048, 0.497030697989493, 0.899896229536556,
 2.15867977892526, 0.443435248995979, 1.60876765916016, 2.26611537562385,
 2.91805305438595, 2.29997207687589, 2.35571703325974, 1.27963227277936,
 0.752385351454657, 1.63630002613963, 0.688251825085977, 2.64643965036121
])
# fixed RIS
STATIC_RS1 = 32
STATIC_RS2 = 32

def wrap_to_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def compute_nearfield(xi, yi, zi, xr, yr, zr, zcut,
                      dr1=-20, dr2=0, randomize=True,
                      fc_mhz=27.2, ant_size_x=5.4, ant_size_y=5.4,
                      qe=1, qf=18,
                      x_range=1000, y_range=1000, x_step=50, y_step=50,
                      verbose=False):
    """
    Compute near-field quantities.
    Returns dict with masks, illumination, XY/YZ cuts, element positions + incident/reflected unit vectors.
    """
    try:
        # force RIS fixed
        ant_x = STATIC_RS1
        ant_y = STATIC_RS2

        # wavenumber (mm)
        lambda0 = 300.0 / fc_mhz   # mm (keeps your original scaling)
        k0 = 2.0 * np.pi / lambda0

        # element positions per MATLAB formula (centered)
        x_vals = ((ant_x - (2*np.arange(1, ant_x+1)-1))*ant_size_x)/2.0   # length ant_x
        y_vals = ((ant_y - (2*np.arange(1, ant_y+1)-1))*ant_size_y)/2.0   # length ant_y
        X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals, indexing='xy')  # shape (ant_y, ant_x)

        # feed/receiver positions
        xf, yf, zf = float(xi), float(yi), float(zi)
        xr_f, yr_f, zr_f = float(xr), float(yr), float(zr)
        rf = np.sqrt(xf*xf + yf*yf + zf*zf)

        # incident vector (unit) from feed toward origin (approx direction from feed)
        incident_vector = np.array([ -xf, -yf, -zf ], dtype=float)
        incident_norm = np.linalg.norm(incident_vector)
        if incident_norm > 0:
            incident_vector = incident_vector / incident_norm
        else:
            incident_vector = np.array([0.0, 0.0, 1.0])

        # reflected vector (unit) pointing to focus/receiver location (from origin to receiver)
        reflected_vector = np.array([ xr_f, yr_f, zr_f ], dtype=float)
        ref_norm = np.linalg.norm(reflected_vector)
        if ref_norm > 0:
            reflected_vector = reflected_vector / ref_norm
        else:
            reflected_vector = np.array([0.0, 0.0, 1.0])

        # feed-element distances
        rf_mn_x = X_mesh - xf
        rf_mn_y = Y_mesh - yf
        rf_mn_z = - zf
        rf_mn = np.sqrt(rf_mn_x**2 + rf_mn_y**2 + rf_mn_z**2)

        # dot products and theta_f_mn
        rf_mn_x_dot = - rf_mn_x * xf
        rf_mn_y_dot = - rf_mn_y * yf
        rf_mn_z_dot = - rf_mn_z * zf
        rf_mn_dot = rf_mn_x_dot + rf_mn_y_dot + rf_mn_z_dot
        rf_mn_new_mod = rf * rf_mn
        with np.errstate(invalid='ignore'):
            theta_f_mn = np.arccos(np.clip(rf_mn_dot / (rf_mn_new_mod + 1e-12), -1.0, 1.0))

        # focus plane and element->focus distances
        z_focus = zr_f
        r_out_mn = np.sqrt( (X_mesh - xr_f)**2 + (Y_mesh - yr_f)**2 + (z_focus)**2 )

        # random phases
        if not randomize:
            phi_ele_rand = np.zeros_like(X_mesh)
        else:
            nels = ant_x * ant_y
            v = phi_ele_rand_sample.copy().ravel()
            if v.size < nels:
                v = np.tile(v, int(np.ceil(nels / v.size)))[:nels]
            else:
                v = v[:nels]
            phi_ele_rand = v.reshape((ant_y, ant_x))

        # compute phi_ph_shift and quantize
        phi_ph_shift = wrap_to_pi( 2.0*np.pi - k0 * (rf_mn + r_out_mn) + phi_ele_rand )
        phi_ph_shift_q = np.where( (phi_ph_shift > (np.pi/2.0)) | (phi_ph_shift < -(np.pi/2.0)), np.pi, 0.0 )
        codeword = (phi_ph_shift_q / np.pi).astype(int)

        # element and feed patterns (central-focus approx)
        with np.errstate(invalid='ignore'):
            theta_emn_center = np.arccos(np.clip(z_focus / (np.sqrt((X_mesh)**2 + (Y_mesh)**2 + z_focus**2) + 1e-12), -1.0, 1.0))
        elem_pat = np.power(np.cos(theta_emn_center), qe)
        f_pat = np.power(np.cos(theta_f_mn), qf)
        with np.errstate(invalid='ignore'):
            theta_out_mn = np.arccos(np.clip(z_focus / (r_out_mn + 1e-12), -1.0, 1.0))
        elem_rad = np.power(np.cos(theta_out_mn), qe)
        denom = (rf_mn * r_out_mn) + 1e-12
        a_mn = (f_pat * elem_pat * elem_rad) / denom

        # per-element illumination magnitude (dB) normalized
        im_db = 20.0 * np.log10(np.abs(a_mn) + 1e-12)
        im_max = np.nanmax(im_db)
        im_norm = im_db - im_max

        # XY cut computation (scan grid)
        x_scan_vals = np.arange(-x_range, x_range + 1, x_step)
        y_scan_vals = np.arange(-y_range, y_range + 1, y_step)
        nx = x_scan_vals.size
        ny = y_scan_vals.size
        E_rad_total = np.zeros((nx, ny), dtype=complex)
        # accumulate element contributions
        # Note: loops are OK for clarity; can be vectorized for speed later
        for m in range(ant_x):
            for n in range(ant_y):
                xm = X_mesh[n, m]
                ym = Y_mesh[n, m]
                rf_mn_val = rf_mn[n, m]
                phi_elem = phi_ele_rand[n, m]
                Xg, Yg = np.meshgrid(x_scan_vals, y_scan_vals, indexing='xy')  # (nx,ny)
                r_out_plane = np.sqrt( (xm - Xg)**2 + (ym - Yg)**2 + (zcut)**2 )
                r_dist = rf_mn_val + r_out_plane
                elem_phase = np.exp(-1j * k0 * r_dist) * np.exp(-1j * phi_ph_shift[n, m])
                a_val = a_mn[n, m]
                E_rad_total += a_val * elem_phase

        rr = 20.0 * np.log10(np.abs(E_rad_total) + 1e-12)
        rr_max = np.nanmax(rr)
        rr_norm = rr - rr_max
        rr_clip = np.clip(rr_norm, dr1, dr2)

        # YZ cut computation
        z_scan_vals = np.arange(0, 2000 + 1, 50)
        y_scan_vals_yz = np.arange(-2000, 2000 + 1, 50)
        nz = z_scan_vals.size
        ny_yz = y_scan_vals_yz.size
        E_rad_yz = np.zeros((nz, ny_yz), dtype=complex)
        x_focus = xr_f
        # accumulate
        for m in range(ant_x):
            for n in range(ant_y):
                xm = X_mesh[n, m]
                ym = Y_mesh[n, m]
                rf_mn_val = rf_mn[n, m]
                Zg, Yg = np.meshgrid(z_scan_vals, y_scan_vals_yz, indexing='xy')  # (ny_yz, nz)
                r_out_plane = np.sqrt( (xm - x_focus)**2 + (ym - Yg)**2 + (Zg)**2 )
                r_dist = rf_mn_val + r_out_plane
                elem_phase = np.exp(-1j * k0 * r_dist) * np.exp(-1j * phi_ph_shift[n, m])
                a_val = a_mn[n, m]
                E_rad_yz += a_val * elem_phase.T

        rr_yz = 20.0 * np.log10(np.abs(E_rad_yz) + 1e-12)
        rr_yz_max = np.nanmax(rr_yz)
        rr_yz_norm = rr_yz - rr_yz_max
        rr_yz_clip = np.clip(rr_yz_norm, dr1, dr2)

        # compute angles (degrees)
        theta_inc = np.arctan2(np.sqrt(xi*xi + yi*yi), zi) if (zi != 0) else 0.0
        phi_inc = np.arctan2(yi, xi) if (xi != 0 or yi != 0) else 0.0
        theta_ref = np.arctan2(np.sqrt(xr_f*xr_f + yr_f*yr_f), zr_f) if (zr_f != 0) else 0.0
        phi_ref = np.arctan2(yr_f, xr_f) if (xr_f != 0 or yr_f != 0) else 0.0

        params = {
            "dr": (dr1, dr2),
            "zcut": zcut,
            "rx": (xr_f, yr_f, zr_f),
            "tx": (xf, yf, zf),
            "theta_inc_deg": np.degrees(theta_inc),
            "phi_inc_deg": np.degrees(phi_inc),
            "theta_ref_deg": np.degrees(theta_ref),
            "phi_ref_deg": np.degrees(phi_ref)
        }

        return {
            "mask": codeword,                         # RS2 x RS1
            "phi_ph_shift_q": phi_ph_shift_q,         # quantized phase (0/pi)
            "phi_ph_shift": phi_ph_shift,
            "im_norm": im_norm,                       # per-element illumination normalized dB
            "rr_norm_xy": rr_norm,                    # normalized dB grid (nx x ny)
            "rr_clip_xy": rr_clip,                    # clipped to DR
            "x_pos": x_scan_vals,
            "y_pos": y_scan_vals,
            "rr_yz_norm": rr_yz_norm,
            "rr_yz_clip": rr_yz_clip,
            "z_pos": z_scan_vals,
            "y_pos_yz": y_scan_vals_yz,
            "x_vals_elements": x_vals,
            "y_vals_elements": y_vals,
            "params": params,
            "incident_vector": incident_vector,
            "reflected_vector": reflected_vector
        }

    except Exception as e:
        tb = traceback.format_exc()
        print("compute_nearfield error:", e)
        print(tb)
        return {"error": str(e), "trace": tb}
    

# Cache near-field compute to speed repeated identical queries from the UI
compute_nearfield = lru_cache(maxsize=64)(compute_nearfield)
        

# ---------- plotting helpers ----------

def build_vector_figure(data, texture_path="grid.png"):
    """
    Build the 3D vector + RIS plane visualization.
    Attempts to load texture_path for plane texture; if missing, shows plain colored plane.
    """
    iv = data.get("incident_vector", np.array([0.0,0.0,1.0]))
    rv = data.get("reflected_vector", np.array([0.0,0.0,1.0]))

    plane_half = 1.3
    ny, nx = STATIC_RS2, STATIC_RS1
    ny = max(ny, 8); nx = max(nx, 8)
    y_vals = np.linspace(-plane_half, plane_half, ny)
    z_vals = np.linspace(-plane_half, plane_half, nx)
    Yp, Zp = np.meshgrid(y_vals, z_vals)
    Xp = np.zeros_like(Yp)

    fig = go.Figure()
    # try to load texture
    tex_present = False
    if os.path.exists(texture_path):
        try:
            img = Image.open(texture_path).convert("RGBA")
            # resize to mesh shape
            img_resized = img.resize((Yp.shape[1], Yp.shape[0]))
            tex = np.asarray(img_resized)
            tex_present = True
        except Exception:
            tex_present = False

    if tex_present:
        fig.add_trace(go.Surface(
            x=Xp, y=Yp, z=Zp,
            surfacecolor=np.ones_like(Xp),
            cmin=0, cmax=1,
            showscale=False, opacity=0.88,
            colorscale=[[0,'rgb(210,190,170)'], [1,'rgb(210,190,170)']],
            hoverinfo='skip',
            name='RIS plane',
            # use custom texture by setting 'colorscale' + 'surfacecolor' trick: use texture as base64 not directly supported
        ))
        # overlay texture as image-like scatter (approx): use marker symbols on plane? Simpler: leave colored plane (texture may not map perfectly)
    else:
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
            marker=dict(size=8, color=color, symbol='diamond'),
            showlegend=False, hoverinfo='skip'
        ))

    # place incident arrow starting at +X side pointing inward (like MATLAB)
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
    mask = np.array(data["mask"])
    mask_deg = np.flipud(mask * 180.0)
    nx = mask.shape[1]; ny = mask.shape[0]
    x = np.arange(1, nx+1)
    y = np.arange(1, ny+1)
    fig = go.Figure(go.Heatmap(z=mask_deg, x=x, y=y,
                               colorscale=[[0.0,"#FFD700"],[0.499,"#FFD700"],[0.5,"#4B0082"],[1.0,"#4B0082"]],
                               zmin=0, zmax=180, showscale=False,
                               hovertemplate="Y=%{x}<br>X=%{y}<br>Phase=%{z}°<extra></extra>"))
    fig.update_layout(title="Randomized Phase Mask", xaxis_title="Y-Elements", yaxis_title="X-Elements", margin=dict(l=60,r=20,t=40,b=50))
    return fig

def build_illumination_figure(data):
    im = np.array(data["im_norm"])
    im_vis = np.flipud(im)
    x = data["x_vals_elements"]
    y = data["y_vals_elements"]
    fig = go.Figure(go.Heatmap(z=im_vis, x=x, y=y, colorscale="Viridis",
                               colorbar=dict(title="Magnitude dB", ticks="outside"),
                               hovertemplate="Y=%{x:.2f} mm<br>X=%{y:.2f} mm<br>dB=%{z:.2f}<extra></extra>"))
    fig.update_layout(title="Illumination Magnitude (per element, normalized dB)", xaxis_title="Y (mm)", yaxis_title="X (mm)", margin=dict(l=60,r=20,t=40,b=50))
    return fig

def build_xy_figure(data):
    rr = np.array(data["rr_clip_xy"])
    x = data["x_pos"]
    y = data["y_pos"]
    fig = go.Figure(go.Heatmap(z=rr.T, x=y, y=x, colorscale="Viridis",
                               colorbar=dict(title="Magnitude (dB)", ticks="outside"),
                               hovertemplate="Y=%{x} mm<br>X=%{y} mm<br>dB=%{z:.2f}<extra></extra>"))
    dr1, dr2 = data.get("params", {}).get("dr", (-20,0))
    fig.data[0].zmin = float(dr1); fig.data[0].zmax = float(dr2)
    zcut = data.get("params", {}).get("zcut", 0)
    zr = data.get("params", {}).get("rx", (None,None,None))[2]
    fig.update_layout(title=f"X-Y Cut at Z = {zcut} mm at Z focus = {zr}", xaxis_title="Y (mm)", yaxis_title="X (mm)", margin=dict(l=60,r=40,t=50,b=50))
    return fig

def build_yz_figure(data):
    rr_yz = np.array(data["rr_yz_clip"])
    zpos = data["z_pos"]
    y = data["y_pos_yz"]
    fig = go.Figure(go.Heatmap(z=rr_yz.T, x=y, y=zpos, colorscale="Viridis",
                               colorbar=dict(title="Magnitude (dB)", ticks="outside"),
                               hovertemplate="Y=%{x} mm<br>Z=%{y} mm<br>dB=%{z:.2f}<extra></extra>"))
    dr1, dr2 = data.get("params", {}).get("dr", (-20,0))
    fig.data[0].zmin = float(dr1); fig.data[0].zmax = float(dr2)
    fig.update_layout(title="Y-Z Cut at X = 0 mm", xaxis_title="Y (mm)", yaxis_title="Z (mm)", margin=dict(l=60,r=40,t=50,b=50))
    return fig
