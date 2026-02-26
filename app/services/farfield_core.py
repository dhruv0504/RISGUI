# app/services/farfield_core.py
# Far-Field compute & plotting service (wrapped from your provided script)

import io
import sys
import traceback
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import functools
from functools import lru_cache

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

        # Compute element contributions in chunks to limit peak memory usage.
        # Use float32/complex64 to reduce memory and speed up operations.
        dtype_f = np.float32
        dtype_c = np.complex64

        kX_f = kX.astype(dtype_f)
        kY_f = kY.astype(dtype_f)
        base_f = base.astype(dtype_f)
        Uv_f = Uv.astype(dtype_f)
        Vv_f = Vv.astype(dtype_f)

        nelem = base_f.size
        nobs = Uv_f.size

        E_vec = np.zeros(nobs, dtype=dtype_c)

        # Chunk size tuned to keep (nelem * chunk) modest; 8192 gives ~8k*1k ~ manageable
        chunk = 8192
        for start in range(0, nobs, chunk):
            end = min(start + chunk, nobs)
            Uc = Uv_f[start:end]
            Vc = Vv_f[start:end]
            # phase_chunk shape: (nelem, chunk)
            phase_chunk = base_f[:, None] + (kX_f[:, None] * Uc[None, :]) + (kY_f[:, None] * Vc[None, :])
            E_chunk = np.sum(np.exp(1j * phase_chunk).astype(dtype_c), axis=0)
            E_vec[start:end] = E_chunk

        E_rad = E_vec.reshape(U.shape)

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


# Cache the far-field compute for repeated identical numeric calls
# Note: arguments must be hashable (numbers/bools). Cache size tuned modestly.
compute_fields = lru_cache(maxsize=64)(compute_fields)

# Plot builders (moved & lightly adapted)

# 

def build_vector_figure_visible(data, texture_rgb=None):
    """
    3D radiation-surface visualization:
      - constructs a parametric spherical surface where radius(θ,φ) ∝ 10^(dB/20)
      - maps clipped dB values (r_clip) to vertex intensity/color
      - overlays RIS plane grid and incident/reflected arrows (from data)
    """


    # extract vectors + grid size
    iv = np.array(data["incident_vector"], dtype=float)
    rv = np.array(data["reflected_vector"], dtype=float)

    try:
        # dB grid (phi x theta) as produced by compute_fields
        r_clip = np.array(data["r_clip"])   # shape (len(phi), len(theta))
        theta_deg = np.array(data["theta"]) # -90..90
        phi_deg = np.array(data["phi"])     # 0..360

        # convert to radians and build param mesh (phi major, theta minor -> same ordering used earlier)
        theta_rad = np.deg2rad(theta_deg)   # shape (len(theta),)
        phi_rad = np.deg2rad(phi_deg)       # shape (len(phi),)
        PH, TH = np.meshgrid(phi_rad, theta_rad, indexing='ij')  # shape (len(phi), len(theta))

        # convert TH (colatitude) from -pi/2..pi/2 to co-latitude 0..pi for spherical mapping:
        # TH is polar angle from XY plane; co_lat = pi/2 - TH
        co_lat = (np.pi / 2.0) - TH

        # convert dB to linear amplitude; small eps to avoid zero
        amp = 10.0 ** (r_clip / 20.0)
        amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)

        # normalize amplitude to [0,1] for visual scaling
        a_min = amp.min() if amp.size else 0.0
        a_max = amp.max() if amp.size else 1.0
        if a_max - a_min < 1e-12:
            amp_norm = np.zeros_like(amp)
        else:
            amp_norm = (amp - a_min) / (a_max - a_min)

        # scale radius: base offset so small sidelobes still visible
        base_radius = 0.2
        scale_radius = 1.1    # overall scale factor for display
        R = base_radius + (amp_norm * scale_radius)

        # spherical -> cartesian coordinates
        Xs = (R * np.sin(co_lat) * np.cos(PH))
        Ys = (R * np.sin(co_lat) * np.sin(PH))
        Zs = (R * np.cos(co_lat))

        # flatten for mesh building
        rows, cols = Xs.shape   # rows=len(phi), cols=len(theta)
        verts_x = Xs.ravel()
        verts_y = Ys.ravel()
        verts_z = Zs.ravel()
        intensity = amp.ravel()   # color mapping by linear amplitude (or could use r_clip.ravel())

        # build triangle faces (two triangles per quad)
        I = []; J = []; K = []
        def idx(i, j): return i * cols + j
        for i in range(rows - 1):
            for j in range(cols - 1):
                a = idx(i, j)
                b = idx(i + 1, j)
                c = idx(i + 1, j + 1)
                d = idx(i, j + 1)
                # triangle a-b-c
                I.append(a); J.append(b); K.append(c)
                # triangle a-c-d
                I.append(a); J.append(c); K.append(d)

        # Build figure
        fig = go.Figure()

        # radiation mesh
        fig.add_trace(go.Mesh3d(
            x=verts_x, y=verts_y, z=verts_z,
            i=I, j=J, k=K,
            intensity=intensity,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Linear amplitude"),
            opacity=0.95,
            flatshading=False,
            name="Radiation surface",
            hoverinfo='skip'
        ))

        # add a translucent RIS plane grid at x=0 for context (small plane)
        plane_half = 1.15
        ny = max(data["X"].shape[0], 8)
        nx = max(data["X"].shape[1], 8)
        y_vals = np.linspace(-plane_half, plane_half, ny)
        z_vals = np.linspace(-plane_half, plane_half, nx)
        for yv in y_vals:
            fig.add_trace(go.Scatter3d(
                x=[0.0, 0.0], y=[yv, yv], z=[z_vals[0], z_vals[-1]],
                mode='lines', line=dict(color='saddlebrown', width=2), showlegend=False, hoverinfo='skip'
            ))
        for zv in z_vals:
            fig.add_trace(go.Scatter3d(
                x=[0.0, 0.0], y=[y_vals[0], y_vals[-1]], z=[zv, zv],
                mode='lines', line=dict(color='saddlebrown', width=2), showlegend=False, hoverinfo='skip'
            ))

        # arrow drawing helper (same style as original)
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

        # place incident arrow coming from +X side (so visible)
        add_arrow_line(fig, [1.5, 0.0, 0.0], -iv, 'red', name='Incident Vector')
        add_arrow_line(fig, [0.0, 0.0, 0.0], rv, 'blue', name='Reflected Vector')

        # Add a central origin marker
        fig.add_trace(go.Scatter3d(x=[0.0], y=[0.0], z=[0.0], mode='markers',
                                   marker=dict(size=4, color='black'), showlegend=False, hoverinfo='skip'))

        fig.update_layout(
            title="3D Radiation Surface (mapped from r_clip)",
            scene=dict(
                xaxis=dict(title='X', backgroundcolor='rgb(245,245,245)', gridcolor='lightgray', zeroline=False),
                yaxis=dict(title='Y', backgroundcolor='rgb(245,245,245)', gridcolor='lightgray', zeroline=False),
                zaxis=dict(title='Z', backgroundcolor='rgb(245,245,245)', gridcolor='lightgray', zeroline=False),
                aspectmode='auto',
                camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9))
            ),
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='white',
            plot_bgcolor='white',
            showlegend=True
        )

        return fig

    except Exception as e:
        # fallback to a simple scene if something fails
        print("build_vector_figure_visible error:", e)
        fallback = go.Figure()
        fallback.update_layout(scene=dict(aspectmode='cube'))
        return fallback


# Public service wrapper expected by tests / Dash
def compute_farfield_pattern(params: dict):
    """
    Lightweight wrapper that computes a far-field pattern and returns a
    dictionary with keys `figure` (plotly Figure) and `pattern` (dB grid).
    Accepts a params dict with optional keys:
      - 'azi' : float (phi incidence in degrees)
      - 'dr_min', 'dr_max' : clipping range in dB
    """
    try:
        phi_inc = float(params.get("azi", 0.0))
    except Exception:
        phi_inc = 0.0

    dr_min = float(params.get("dr_min", -60.0))
    dr_max = float(params.get("dr_max", 0.0))

    data = compute_fields(
        theta_inc_deg=0.0,
        phi_inc_deg=phi_inc,
        theta_ref_deg=0.0,
        phi_ref_deg=0.0,
        RS1_arg=STATIC_RS1,
        RS2_arg=STATIC_RS2,
        DR1=dr_min,
        DR2=dr_max,
        randomize=False,
    )

    fig = build_vector_figure_visible(data)
    pattern = data.get("r_clip") if isinstance(data, dict) else None
    return {"figure": fig, "pattern": pattern}


# def build_vector_figure_visible(data, texture_rgb=None):
    """
    Replacement that renders:
      - textured rectangular panel (from /mnt/data/ff74ec77-1323-4910-818f-cf2fb6151f68.png)
      - several glossy parametric lobes (Mesh3d) to mimic the decorative pattern
      - original grid lines and incident/reflected arrows

    Note: This is a stylized approximation — Plotly doesn't support full photorealistic
    texture-mapped materials (no true UV-mapping / PBR), so the result is an artistic
    3D composition that keeps the domain meaning (panel + vectors).
    """


    iv = np.array(data["incident_vector"], dtype=float)
    rv = np.array(data["reflected_vector"], dtype=float)

    # scene / plane sizing
    plane_half = 1.3
    ny, nx = data["X"].shape
    ny = max(ny, 8); nx = max(nx, 8)

    # Make the textured panel (vertical rectangle) positioned near x ~ 0.6
    panel_center_x = 0.6
    panel_w = 1.6   # panel width (y span)
    panel_h = 1.6   # panel height (z span)
    py = np.linspace(-panel_w/2, panel_w/2, nx)
    pz = np.linspace(-panel_h/2, panel_h/2, ny)
    PY, PZ = np.meshgrid(py, pz)   # shape (ny, nx)
    PX = np.ones_like(PY) * panel_center_x

    fig = go.Figure()

    # Try to texture the panel using the provided PNG path
    img_path = "/mnt/data/ff74ec77-1323-4910-818f-cf2fb6151f68.png"
    try:
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGBA")
            # Resize to mesh dimensions (nx x ny)
            img_resized = img.resize((nx, ny), Image.LANCZOS)
            rgba = np.array(img_resized)  # (ny, nx, 4)
            rgb = rgba[..., :3]
            # Flatten to build colorscale (one color per pixel)
            flat_rgb = rgb.reshape(-1, 3)
            ncolors = flat_rgb.shape[0]
            colorscale = []
            for i, (r, g, b) in enumerate(flat_rgb):
                pos = i / max(ncolors - 1, 1)
                colorscale.append((pos, f"rgb({int(r)},{int(g)},{int(b)})"))
            # Build surfacecolor normalized 0..1
            idxs = np.arange(ncolors).reshape((ny, nx))
            surfacecolor = idxs.astype(np.float32) / max(ncolors - 1, 1)
            # Optionally flip to match orientation visually
            surfacecolor = np.flipud(surfacecolor)

            fig.add_trace(go.Surface(
                x=PX, y=PY, z=PZ,
                surfacecolor=surfacecolor,
                colorscale=colorscale,
                cmin=0.0, cmax=1.0,
                showscale=False, opacity=0.98,
                name='Decorative panel',
                hoverinfo='skip'
            ))
        else:
            # If image missing, fallback to a warm panel color
            raise FileNotFoundError("Texture image missing")
    except Exception as e:
        # Fallback plain panel
        print("Texture panel error:", e); sys.stdout.flush()
        fig.add_trace(go.Surface(
            x=PX, y=PY, z=PZ,
            surfacecolor=np.ones_like(PX),
            showscale=False, opacity=0.94,
            colorscale=[[0,'rgb(250,235,200)'], [1,'rgb(220,180,120)']],
            hoverinfo='skip', name='Panel'
        ))

    # Function to generate a glossy lobe / petal mesh (parametric)
    def make_lobe(center, direction, scale=1.0, petals=6, smooth=48, length=1.0, thickness=0.8):
        """
        Create a petal/lobe-like mesh around 'direction' vector.
        - center: 3-vector offset for the lobe
        - direction: unit 3-vector pointing toward main axis of lobe
        """
        # Ensure direction unit
        d = np.array(direction, dtype=float)
        dnorm = np.linalg.norm(d)
        if dnorm < 1e-8:
            d = np.array([0.0, 0.0, 1.0])
        else:
            d = d / dnorm

        # Build orthonormal basis (u,v,w) with w=d
        # pick arbitrary vector not parallel to d
        a = np.array([1.0, 0.1, 0.2])
        if abs(np.dot(a, d)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        u = np.cross(d, a)
        u /= np.linalg.norm(u)
        v = np.cross(d, u)

        # param space
        theta = np.linspace(0, 2*np.pi, smooth)
        phi = np.linspace(0, np.pi, smooth//2)
        th, ph = np.meshgrid(theta, phi)

        # radial variation to get petals: r = 1 + 0.4*cos(petals*theta) * sin(phi)^power
        pet = 0.55 * np.cos(petals * th) * (np.sin(ph) ** 2.2)
        r = 0.8 + pet
        # shape in spherical coords: radius scaled by an envelope so it tapers
        radius = (length * (np.sin(ph) ** 1.2) * r) * scale

        # spherical to cartesian in local basis (w points outward)
        Xc = (radius * np.sin(ph) * np.cos(th))[:, :]
        Yc = (radius * np.sin(ph) * np.sin(th))[:, :]
        Zc = (radius * np.cos(ph))[:, :]

        # Compose points into world coords P = center + u*Xc + v*Yc + d*Zc
        pts = center[None, None, :] + (u[None, None, :] * Xc[..., None] +
                                       v[None, None, :] * Yc[..., None] +
                                       d[None, None, :] * Zc[..., None])
        # Flatten grid into triangles for Mesh3d
        rows, cols, _ = pts.shape
        verts = pts.reshape((-1, 3))
        # build faces (i,j,k) using grid indices
        def idx(i, j): return i * cols + j
        I = []
        J = []
        K = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                a_idx = idx(i, j)
                b_idx = idx(i + 1, j)
                c_idx = idx(i + 1, j + 1)
                d_idx = idx(i, j + 1)
                # two triangles: a-b-c and a-c-d
                I += [a_idx, a_idx]
                J += [b_idx, c_idx]
                K += [c_idx, d_idx]
        # color per vertex (gloss effect): use simple gradient based on Zc (height)
        z_vals = verts[:, 2]
        zmin, zmax = z_vals.min(), z_vals.max()
        if zmax - zmin < 1e-6:
            z_norm = np.zeros_like(z_vals)
        else:
            z_norm = (z_vals - zmin) / (zmax - zmin)
        # map to rgb-like (bluish to cyan to white)
        colors = ["rgb({},{},{})".format(
            int(20 + 200*z), int(140 + 100*z), int(200 + 55*z)
        ) for z in z_norm]
        return verts[:, 0], verts[:, 1], verts[:, 2], I, J, K, colors

    # Create a few lobes:
    # Largest lobe pointing along 'rv' (reflected vector) and placed slightly behind the panel
    verts_x, verts_y, verts_z, I, J, K, colors = make_lobe(
        center=np.array([-0.1, 0.0, 0.0]), direction=rv,
        scale=1.0, petals=5, smooth=64, length=1.05
    )
    fig.add_trace(go.Mesh3d(
        x=verts_x, y=verts_y, z=verts_z,
        i=I, j=J, k=K,
        facecolor=colors,
        flatshading=False,
        name="Lobe (reflected)",
        opacity=0.95,
        hoverinfo='skip',
        showscale=False
    ))

    # Secondary lobe pointing roughly opposite (incident)
    verts_x2, verts_y2, verts_z2, I2, J2, K2, colors2 = make_lobe(
        center=np.array([0.6, -0.2, 0.0]), direction=-iv,
        scale=0.7, petals=4, smooth=48, length=0.85
    )
    fig.add_trace(go.Mesh3d(
        x=verts_x2, y=verts_y2, z=verts_z2,
        i=I2, j=J2, k=K2,
        facecolor=colors2,
        flatshading=False,
        name="Lobe (incident)",
        opacity=0.92,
        hoverinfo='skip',
        showscale=False
    ))

    # Small glossy highlights (spherical blobs) around origin for visual richness
    def add_highlight(center, r=0.12, segments=20):
        u = np.linspace(0, 2*np.pi, segments)
        v = np.linspace(0, np.pi, segments//2)
        uu, vv = np.meshgrid(u, v)
        Xs = center[0] + r * np.cos(uu) * np.sin(vv)
        Ys = center[1] + r * np.sin(uu) * np.sin(vv)
        Zs = center[2] + r * np.cos(vv)
        verts = np.stack([Xs.ravel(), Ys.ravel(), Zs.ravel()], axis=1)
        rows, cols = Xs.shape
        I = []; J = []; K = []
        def idx2(i,j): return i*cols + j
        for i in range(rows-1):
            for j in range(cols-1):
                a=idx2(i,j); b=idx2(i+1,j); c=idx2(i+1,j+1); d=idx2(i,j+1)
                I+= [a,a]; J+=[b,c]; K+=[c,d]
        colors = ["rgb(240,240,255)" for _ in range(verts.shape[0])]
        return verts[:,0], verts[:,1], verts[:,2], I, J, K, colors

    hx, hy, hz, HI, HJ, HK, Hcols = add_highlight(np.array([0.0, 0.0, 0.0]), r=0.08, segments=16)
    fig.add_trace(go.Mesh3d(x=hx, y=hy, z=hz, i=HI, j=HJ, k=HK, facecolor=Hcols, opacity=0.95, hoverinfo='skip', showscale=False))

    # grid lines on the RIS plane area (for context) - reuse your y_vals/z_vals approach but oriented on x=0 plane
    nyg = max(8, data["X"].shape[0])
    nxg = max(8, data["X"].shape[1])
    y_vals = np.linspace(-plane_half, plane_half, nyg)
    z_vals = np.linspace(-plane_half, plane_half, nxg)
    for yv in y_vals:
        fig.add_trace(go.Scatter3d(x=[0,0], y=[yv,yv], z=[z_vals[0], z_vals[-1]],
                                   mode='lines', line=dict(color='saddlebrown', width=2), showlegend=False, hoverinfo='skip'))
    for zv in z_vals:
        fig.add_trace(go.Scatter3d(x=[0,0], y=[y_vals[0], y_vals[-1]], z=[zv,zv],
                                   mode='lines', line=dict(color='saddlebrown', width=2), showlegend=False, hoverinfo='skip'))

    # Arrow helper (exactly like your original)
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

    # Final layout styling
    fig.update_layout(
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1.0, y=1.0, z=0.85),
            xaxis=dict(range=[-2,2], title='X', backgroundcolor='rgb(245,245,245)', gridcolor='lightgray'),
            yaxis=dict(range=[-2,2], title='Y', backgroundcolor='rgb(245,245,245)', gridcolor='lightgray'),
            zaxis=dict(range=[-2,2], title='Z', backgroundcolor='rgb(245,245,245)', gridcolor='lightgray'),
            camera=dict(eye=dict(x=1.3, y=-1.4, z=0.9))
        ),
        margin=dict(l=8, r=8, t=28, b=8),
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
