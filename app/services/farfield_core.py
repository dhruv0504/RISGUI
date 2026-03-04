# app/services/farfield_core.py
# Far-Field compute & plotting service (wrapped from your provided script)

# Standard I/O utilities (not heavily used here but available)
import io
import sys

# Used to capture full exception stack traces
import traceback

# Numerical computing library (arrays, math, vectorization)
import numpy as np

# Plotly for interactive 2D/3D visualization
import plotly.graph_objects as go

# PIL image module (not directly used in this snippet but imported)
from PIL import Image

# Functional programming utilities
import functools

# LRU cache decorator for memoizing compute_fields results
from functools import lru_cache

# Small helpers

# Wraps angle(s) in radians into range [-π, π]
# Ensures phase continuity and prevents overflow
def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

# sample random vector from your paste
# Predefined random phase sample used for per-element phase randomization
phi_ele_rand_sample = np.array([
 0.219827130698740, 2.98461967174048, 0.497030697989493, 0.899896229536556,
 2.15867977892526, 0.443435248995979, 1.60876765916016, 2.26611537562385,
 2.91805305438595, 2.29997207687589, 2.35571703325974, 1.27963227277936,
 0.752385351454657, 1.63630002613963, 0.688251825085977, 2.64643965036121
])

# Default static RIS
# Number of columns (x direction)
STATIC_RS1 = 32
# Number of rows (y direction)
STATIC_RS2 = 32

# Core electromagnetic far-field computation
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
        # Protect fc from zero / falsy values to avoid division by zero
        if not fc:
            fc = 28.5

        # Force static RIS size (override passed RS1_arg/RS2_arg)
        # RS1 = number of elements in x-direction
        RS1 = 32  # columns (x)
        # RS2 = number of elements in y-direction
        RS2 = 32  # rows    (y)

        # Convert angles from degrees to radians for trigonometric functions
        ti = np.deg2rad(theta_inc_deg)
        pi = np.deg2rad(phi_inc_deg)
        tr = np.deg2rad(theta_ref_deg)
        pr = np.deg2rad(phi_ref_deg)

        # Build incident unit direction vector in Cartesian coordinates
        incident_vector = np.array([np.sin(ti) * np.cos(pi),
                                    np.sin(ti) * np.sin(pi),
                                    np.cos(ti)])

        # Build reflected (desired beam) unit direction vector
        reflected_vector = np.array([np.sin(tr) * np.cos(pr),
                                     np.sin(tr) * np.sin(pr),
                                     np.cos(tr)])

        # Wavelength calculation: lambda = c / f
        # Here 300 corresponds to speed of light in mm/ns units
        lambda0 = 300.0 / fc

        # Wavenumber k = 2π / λ
        k0 = 2.0 * np.pi / lambda0

        print("\n=== Fundamental Parameters ===")
        print("Frequency (fc):", fc)
        print("Wavelength (lambda0):", lambda0)
        print("Wavenumber (k0):", k0)


        # Total physical size of RIS in x and y
        R_x = ant_size_x * RS1
        R_y = ant_size_y * RS2

        # Element center coordinates in x-direction
        x_vals = np.linspace(-R_x/2 + ant_size_x/2, R_x/2 - ant_size_x/2, RS1)

        # Element center coordinates in y-direction
        y_vals = np.linspace(-R_y/2 + ant_size_y/2, R_y/2 - ant_size_y/2, RS2)

        # Create 2D coordinate grid of element positions
        X, Y = np.meshgrid(x_vals, y_vals)   # shape (RS2, RS1)

        print("\n=== RIS Geometry ===")
        print("RIS size:", RS1, "x", RS2)
        print("Element spacing X:", ant_size_x)
        print("Element spacing Y:", ant_size_y)
        print("X shape:", X.shape)

        # Compute incident wavevector components
        kx_feed = k0 * np.sin(ti) * np.cos(pi)
        ky_feed = k0 * np.sin(ti) * np.sin(pi)

        # Compute incident phase at each RIS element
        kr = wrap_to_pi(kx_feed * X + ky_feed * Y)

        # Element randomization phase (per-element)
        if not randomize:
            # If randomization disabled, zero phase offset
            phi_ele_rand = np.zeros_like(X)
        else:
            # Total number of elements
            nels = RS1 * RS2
            # Flatten random sample
            v = phi_ele_rand_sample.copy().ravel()
            # If sample too small, tile it
            if v.size < nels:
                v = np.tile(v, int(np.ceil(nels / v.size)))[:nels]
            else:
                v = v[:nels]
            # Reshape to RIS grid
            phi_ele_rand = v.reshape((RS2, RS1))

        # Beam (reflected) wavevector components
        kx_beam = k0 * np.sin(tr) * np.cos(pr)
        ky_beam = k0 * np.sin(tr) * np.sin(pr)

        # Desired reflected phase at each element
        k_beam_r = wrap_to_pi(-(kx_beam * X + ky_beam * Y))

        # Required phase shift to steer beam
        phase_shift = wrap_to_pi(k_beam_r - kr - phi_ele_rand)

        # 1-bit quantization: 0 or π
        phase_shift_q = np.pi * (np.abs(phase_shift) >= (np.pi / 2))

        # Binary mask (0 or 1)
        mask = phase_shift_q / np.pi   # 0 or 1

        print("\n=== Phase Mask ===")
        print("Mask unique values:", np.unique(mask))
        print("Number of 0 states:", np.sum(mask == 0))
        print("Number of 1 states:", np.sum(mask == 1))

        # Observation grid angles
        theta = np.arange(-90, 91, 1)   # elevation angles
        phi = np.arange(0, 361, 1)      # azimuth angles

        # Convert observation grid to radians
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)

        # Create angular meshgrid
        PHI, THETA = np.meshgrid(phi_rad, theta_rad, indexing='ij')

        # Direction cosines (u,v) representation
        U = np.sin(THETA) * np.cos(PHI)
        V = np.sin(THETA) * np.sin(PHI)

        # Static total phase at each element
        static_phase = kr + phase_shift_q + phi_ele_rand

        # Flatten element positions
        Xv = X.ravel()
        Yv = Y.ravel()

        # Flatten phase array
        base = static_phase.ravel()

        # Precompute kX and kY products
        kX = k0 * Xv
        kY = k0 * Yv

        # Flatten observation direction cosines
        Uv = U.ravel()
        Vv = V.ravel()

        # Use lower precision for memory/performance optimization
        dtype_f = np.float32
        dtype_c = np.complex64

        # Convert arrays to float32
        kX_f = kX.astype(dtype_f)
        kY_f = kY.astype(dtype_f)
        base_f = base.astype(dtype_f)
        Uv_f = Uv.astype(dtype_f)
        Vv_f = Vv.astype(dtype_f)

        # Number of elements
        nelem = base_f.size
        # Number of observation points
        nobs = Uv_f.size

        # Initialize far-field complex vector
        E_vec = np.zeros(nobs, dtype=dtype_c)

        # Chunk processing to avoid huge memory allocation
        chunk = 8192

        # Loop through observation points in chunks
        for start in range(0, nobs, chunk):
            end = min(start + chunk, nobs)
            Uc = Uv_f[start:end]
            Vc = Vv_f[start:end]

            # Compute phase for each element and observation
            phase_chunk = base_f[:, None] + (kX_f[:, None] * Uc[None, :]) + (kY_f[:, None] * Vc[None, :])

            # Sum complex exponentials across elements
            E_chunk = np.sum(np.exp(1j * phase_chunk).astype(dtype_c), axis=0)

            # Store results
            E_vec[start:end] = E_chunk

        # Reshape to angular grid
        E_rad = E_vec.reshape(U.shape)

        print("\n=== Field Computation ===")
        print("Number of elements:", nelem)
        print("Number of observation points:", nobs)
        print("E_rad shape:", E_rad.shape)

        # Avoid log(0)
        eps = 1e-12

        # Convert magnitude to dB
        r = 20.0 * np.log10(np.abs(E_rad) + eps)

        # Normalize so max is 0 dB
        r_norm = r - np.max(r)

        # Clip to dynamic range
        r_clip = np.clip(r_norm, DR1, DR2)

        print("\n=== Radiation Pattern ===")
        print("Max dB (should be 0):", np.max(r_norm))
        print("Min dB:", np.min(r_norm))
        print("Clipped range:", DR1, "to", DR2)

        # Return all relevant data
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
        # Capture full traceback on failure
        tb = traceback.format_exc()
        print("compute_fields error:", e)
        print(tb)
        return {"error": str(e), "trace": tb}


# Cache identical compute calls for speed
compute_fields = lru_cache(maxsize=64)(compute_fields)

def build_vector_figure_visible(data, texture_rgb=None):
    """
    3D radiation-surface visualization:
      - constructs a parametric spherical surface where radius(θ,φ) ∝ 10^(dB/20)
      - maps clipped dB values (r_clip) to vertex intensity/color
      - overlays RIS plane grid and incident/reflected arrows (from data)
    """

    # Extract incident and reflected vectors from data dictionary
    # Convert to float numpy arrays to ensure numeric operations work properly
    iv = np.array(data["incident_vector"], dtype=float)
    rv = np.array(data["reflected_vector"], dtype=float)

    try:
        # Extract clipped dB radiation grid (phi x theta)
        r_clip = np.array(data["r_clip"])   # shape (len(phi), len(theta))

        # Extract angular coordinate arrays in degrees
        theta_deg = np.array(data["theta"]) # -90..90
        phi_deg = np.array(data["phi"])     # 0..360

        # Convert angular arrays to radians for trigonometric operations
        theta_rad = np.deg2rad(theta_deg)   # shape (len(theta),)
        phi_rad = np.deg2rad(phi_deg)       # shape (len(phi),)

        # Create 2D angular meshgrid (phi-major indexing)
        PH, TH = np.meshgrid(phi_rad, theta_rad, indexing='ij')  # shape (len(phi), len(theta))

        # Convert elevation angle (measured from XY-plane)
        # into co-latitude (measured from +Z axis)
        # Needed for proper spherical coordinate mapping
        co_lat = (np.pi / 2.0) - TH

        # Convert dB values to linear amplitude
        # amplitude = 10^(dB/20)
        amp = 10.0 ** (r_clip / 20.0)

        # Replace NaN or infinite values with 0
        amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)

        # Determine min/max amplitude for normalization
        a_min = amp.min() if amp.size else 0.0
        a_max = amp.max() if amp.size else 1.0

        # Prevent divide-by-zero during normalization
        if a_max - a_min < 1e-12:
            amp_norm = np.zeros_like(amp)
        else:
            # Normalize amplitude to range [0,1]
            amp_norm = (amp - a_min) / (a_max - a_min)

        # Base radius so even small sidelobes remain visible
        base_radius = 0.2

        # Overall scaling factor for visual size of radiation pattern
        scale_radius = 1.1

        # Final radius at each angular point
        R = base_radius + (amp_norm * scale_radius)

        # Convert spherical coordinates to Cartesian coordinates
        # X = R sin(θ) cos(φ)
        # Y = R sin(θ) sin(φ)
        # Z = R cos(θ)
        Xs = (R * np.sin(co_lat) * np.cos(PH))
        Ys = (R * np.sin(co_lat) * np.sin(PH))
        Zs = (R * np.cos(co_lat))

        # Get mesh dimensions
        rows, cols = Xs.shape   # rows=len(phi), cols=len(theta)

        # Flatten coordinate arrays into 1D vertex lists
        verts_x = Xs.ravel()
        verts_y = Ys.ravel()
        verts_z = Zs.ravel()

        # Use linear amplitude as color intensity for surface
        intensity = amp.ravel()

        # Initialize triangle index lists for Mesh3d
        I = []; J = []; K = []

        # Helper function converting 2D grid index to flattened index
        def idx(i, j): return i * cols + j

        # Build triangular faces (2 triangles per quad)
        for i in range(rows - 1):
            for j in range(cols - 1):

                # Get four corner indices of quad
                a = idx(i, j)
                b = idx(i + 1, j)
                c = idx(i + 1, j + 1)
                d = idx(i, j + 1)

                # First triangle (a-b-c)
                I.append(a); J.append(b); K.append(c)

                # Second triangle (a-c-d)
                I.append(a); J.append(c); K.append(d)

        # Create empty Plotly figure
        fig = go.Figure()

        # Add 3D radiation surface mesh
        fig.add_trace(go.Mesh3d(
            x=verts_x, y=verts_y, z=verts_z,   # vertex positions
            i=I, j=J, k=K,                     # triangle connectivity
            intensity=intensity,               # color mapping
            colorscale='Viridis',              # colormap
            showscale=True,                    # show colorbar
            colorbar=dict(title="Linear amplitude"),
            opacity=0.95,                      # slight transparency
            flatshading=False,                 # smooth shading
            name="Radiation surface",
            hoverinfo='skip'
        ))

        # Define half-width of RIS plane grid (visual reference)
        plane_half = 1.15

        # Determine grid resolution based on RIS element count
        ny = max(data["X"].shape[0], 8)
        nx = max(data["X"].shape[1], 8)

        # Create evenly spaced grid lines
        y_vals = np.linspace(-plane_half, plane_half, ny)
        z_vals = np.linspace(-plane_half, plane_half, nx)

        # Draw horizontal lines (parallel to z-axis)
        for yv in y_vals:
            fig.add_trace(go.Scatter3d(
                x=[0.0, 0.0], y=[yv, yv], z=[z_vals[0], z_vals[-1]],
                mode='lines',
                line=dict(color='saddlebrown', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Draw vertical lines (parallel to y-axis)
        for zv in z_vals:
            fig.add_trace(go.Scatter3d(
                x=[0.0, 0.0], y=[y_vals[0], y_vals[-1]], z=[zv, zv],
                mode='lines',
                line=dict(color='saddlebrown', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Helper function to draw a 3D arrow
        def add_arrow_line(fig, start, vec, color, name=None):

            # Ensure numeric numpy arrays
            start = np.array(start, dtype=float)
            vec = np.array(vec, dtype=float)

            # Compute vector magnitude
            norm = np.linalg.norm(vec)

            # If zero vector, replace with default Z-direction
            if norm < 1e-9:
                vec = np.array([0.,0.,1.])
                norm = 1.0

            # Normalize direction
            dir_u = vec / norm

            # Visible arrow length
            vis_len = 1.0

            # Compute arrow tip
            tip = start + dir_u * vis_len

            # Compute shortened line portion (for arrow shaft)
            line_end = start + dir_u * (vis_len * 0.86)

            # Draw arrow shaft
            fig.add_trace(go.Scatter3d(
                x=[start[0], line_end[0]],
                y=[start[1], line_end[1]],
                z=[start[2], line_end[2]],
                mode='lines',
                line=dict(color=color, width=6),
                name=name,
                hoverinfo='skip'
            ))

            # Draw arrow head marker
            fig.add_trace(go.Scatter3d(
                x=[tip[0]], y=[tip[1]], z=[tip[2]],
                mode='markers',
                marker=dict(size=8, color=color, symbol='diamond'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Draw incident vector arrow (coming from +X direction)
        add_arrow_line(fig, [1.5, 0.0, 0.0], -iv, 'red', name='Incident Vector')

        # Draw reflected vector arrow (from origin outward)
        add_arrow_line(fig, [0.0, 0.0, 0.0], rv, 'blue', name='Reflected Vector')

        # Add origin marker
        fig.add_trace(go.Scatter3d(
            x=[0.0], y=[0.0], z=[0.0],
            mode='markers',
            marker=dict(size=4, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Configure 3D scene layout
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

        print("\n=== 3D Surface Stats ===")
        print("Amplitude min:", amp.min())
        print("Amplitude max:", amp.max())
        print("Vertices:", len(verts_x))

        # Return fully constructed figure
        return fig

    except Exception as e:
        # If error occurs, print message
        print("build_vector_figure_visible error:", e)

        # Create simple fallback 3D figure
        fallback = go.Figure()
        fallback.update_layout(scene=dict(aspectmode='cube'))

        # Return fallback figure
        return fallback
    
# Public service wrapper expected by tests / Dash
# This function acts as a lightweight interface layer between:
#   - Frontend (Dash app / API)
#   - Core electromagnetic computation (compute_fields)
def compute_farfield_pattern(params: dict):
    """
    Lightweight wrapper that computes a far-field pattern and returns a
    dictionary with keys `figure` (plotly Figure) and `pattern` (dB grid).
    Accepts a params dict with optional keys:
      - 'azi' : float (phi incidence in degrees)
      - 'dr_min', 'dr_max' : clipping range in dB
    """

    # Try to extract azimuth incidence angle from params dictionary
    # If missing or invalid, default to 0.0 degrees
    try:
        phi_inc = float(params.get("azi", 0.0))
    except Exception:
        # Fallback in case conversion fails
        phi_inc = 0.0

    # Extract dynamic range minimum (lower clipping bound in dB)
    # Default is -60 dB
    dr_min = float(params.get("dr_min", -60.0))

    # Extract dynamic range maximum (upper clipping bound in dB)
    # Default is 0 dB
    dr_max = float(params.get("dr_max", 0.0))

    # Call the core far-field computation function
    # Fixed configuration:
    #   - Incident elevation = 0°
    #   - Reflected direction = broadside (0°, 0°)
    #   - Static RIS size (32x32)
    #   - Randomization disabled
    data = compute_fields(
        theta_inc_deg=0.0,           # incident elevation
        phi_inc_deg=phi_inc,         # incident azimuth from params
        theta_ref_deg=0.0,           # reflection elevation
        phi_ref_deg=0.0,             # reflection azimuth
        RS1_arg=STATIC_RS1,          # unused (forced internally)
        RS2_arg=STATIC_RS2,          # unused (forced internally)
        DR1=dr_min,                  # lower dB clipping
        DR2=dr_max,                  # upper dB clipping
        randomize=False,             # no random element phase
    )

    # Build 3D radiation visualization from computed data
    fig = build_vector_figure_visible(data)

    # Extract clipped radiation pattern grid if computation succeeded
    # Otherwise return None
    pattern = data.get("r_clip") if isinstance(data, dict) else None

    # Return dictionary expected by frontend / tests
    #   "figure"  → interactive Plotly 3D radiation surface
    #   "pattern" → 2D numpy array of clipped dB values
    return {"figure": fig, "pattern": pattern}
def build_phase_figure(data):

    # Extract the 1-bit phase mask (values are 0 or 1)
    # Shape: (RS2 rows, RS1 columns)
    mask = np.array(data["mask"])   # shape (RS2, RS1)

    # Extract x-axis element coordinates (RIS horizontal positions)
    x = data["x_vals"]

    # Extract y-axis element coordinates (RIS vertical positions)
    y = data["y_vals"]

    # Convert binary mask (0 or 1) into phase degrees (0° or 180°)
    # Since 1-bit RIS uses only two phase states: 0 and π
    mask_deg = mask * 180.0

    # Flip vertically so orientation matches physical RIS layout
    # (Meshgrid origin vs plotting origin difference correction)
    mask_deg = np.flipud(mask_deg)

    # Create Plotly heatmap figure
    fig = go.Figure(
        go.Heatmap(

            # Z-values represent phase in degrees (0° or 180°)
            z=mask_deg,

            # X-axis positions (horizontal axis)
            x=x,

            # Y-axis positions (vertical axis)
            y=y,

            # Custom two-color scale:
            #   0°   → Gold
            #   180° → Indigo
            colorscale=[
                [0.0, "#FFD700"], [0.499, "#FFD700"],
                [0.5, "#4B0082"], [1.0, "#4B0082"]
            ],

            # Minimum phase value
            zmin=0,

            # Maximum phase value
            zmax=180,

            # Hide color bar (since only two discrete values)
            showscale=False
        )
    )

    # Configure layout properties
    fig.update_layout(

        # Title shown above heatmap
        title="Randomized Phase Mask",

        # X-axis label
        xaxis_title="y-axis (wavelength)",

        # Y-axis label
        yaxis_title="z-axis (wavelength)",

        # Force equal aspect ratio (square elements)
        yaxis_scaleanchor="x",

        # Margins around plot
        margin=dict(l=60, r=20, t=40, b=50)
    )

    # Return fully constructed phase heatmap figure
    return fig

def build_beam_figure(data, DR1, DR2, grid_size=480):

    # Extract clipped radiation pattern (dB values)
    # Shape: (len(phi), len(theta))
    r = np.array(data["r_clip"])        # shape (len(phi), len(theta))

    # Convert stored theta angles from degrees to radians
    theta = np.deg2rad(data["theta"])

    # Convert stored phi angles from degrees to radians
    phi = np.deg2rad(data["phi"])

    # Create angular meshgrid
    # TH and PH will match radiation grid dimensions
    TH, PH = np.meshgrid(theta, phi)   # TH,PH shape = (len(phi), len(theta))

    # Convert spherical angles to direction cosines
    # u = sin(theta) cos(phi)
    u = np.sin(TH) * np.cos(PH)

    # v = sin(theta) sin(phi)
    v = np.sin(TH) * np.sin(PH)

    # Flatten arrays for easier indexing
    u_flat = u.ravel()
    v_flat = v.ravel()
    r_flat = r.ravel()

    # Keep only points inside visible unit circle (u² + v² ≤ 1)
    # This corresponds to physically valid radiation directions
    inside = (u_flat**2 + v_flat**2) <= 1.0

    # Filter valid direction cosine points
    u_pts = u_flat[inside]
    v_pts = v_flat[inside]
    r_pts = r_flat[inside]

    # Create uniform grid for final 2D heatmap (v horizontal, u vertical)
    xs = np.linspace(-1.0, 1.0, grid_size)   # v axis (horizontal)
    ys = np.linspace(-1.0, 1.0, grid_size)   # u axis (vertical)

    # Initialize grid with NaN values
    Z = np.full((grid_size, grid_size), np.nan, dtype=float)

    # Map continuous v values into grid x-indices
    idx_x = np.clip(
        np.round(((v_pts + 1.0) / 2.0) * (grid_size - 1)).astype(int),
        0, grid_size-1
    )

    # Map continuous u values into grid y-indices
    idx_y = np.clip(
        np.round(((u_pts + 1.0) / 2.0) * (grid_size - 1)).astype(int),
        0, grid_size-1
    )

    # Populate grid with maximum dB value per pixel
    # If multiple angular points fall into same grid cell,
    # keep strongest value (better visual main lobe representation)
    for xi, yi, rv in zip(idx_x, idx_y, r_pts):
        cur = Z[yi, xi]
        if np.isnan(cur) or (rv > cur):
            Z[yi, xi] = rv

    # Fill small NaN holes by averaging neighbors (simple smoothing)
    for _ in range(2):

        # Identify NaN cells
        nan_mask = np.isnan(Z)

        # If no NaNs remain, stop early
        if not nan_mask.any():
            break

        # Shift grid in 4 directions to collect neighbors
        up = np.roll(Z, -1, axis=0)
        down = np.roll(Z, 1, axis=0)
        left = np.roll(Z, 1, axis=1)
        right = np.roll(Z, -1, axis=1)

        # Stack neighbors and compute sum ignoring NaNs
        neighbor_sum = np.nansum(
            np.stack([up, down, left, right], axis=0),
            axis=0
        )

        # Count valid neighbors per cell
        neighbor_count = np.sum(
            ~np.isnan(np.stack([up, down, left, right], axis=0)),
            axis=0
        )

        # Compute average of valid neighbors
        fill_vals = np.where(
            neighbor_count > 0,
            neighbor_sum / neighbor_count,
            np.nan
        )

        # Replace NaN cells with averaged values where possible
        Z = np.where(
            nan_mask & (~np.isnan(fill_vals)),
            fill_vals,
            Z
        )

    # Create meshgrid for circular mask
    XV, YU = np.meshgrid(xs, ys)

    # Define circular boundary (visible hemisphere)
    circle_mask = (XV**2 + YU**2) <= 1.0

    # Apply circular mask (outside circle → NaN)
    Z_masked = np.where(circle_mask, Z, np.nan)

    print("\n=== Beam Plot Stats ===")
    print("Grid size:", grid_size)
    print("Valid beam points:", np.sum(~np.isnan(Z_masked)))
    print("Max beam dB:", np.nanmax(Z_masked))
    print("Min beam dB:", np.nanmin(Z_masked))

    # Create 2D heatmap figure
    fig = go.Figure(
        go.Heatmap(
            x=xs,
            y=ys,
            z=Z_masked,

            # Colormap for magnitude
            colorscale="Viridis",

            # Apply clipping range
            zmin=DR1,
            zmax=DR2,

            # Configure colorbar
            colorbar=dict(title="Magnitude dB", ticks="outside"),

            # Custom hover display
            hovertemplate="v=%{x:.3f}<br>u=%{y:.3f}<br>dB=%{z:.2f}<extra></extra>"
        )
    )

    # Create circular outline (unit circle boundary)
    ang = np.linspace(0, 2*np.pi, 512)
    circ_x = np.cos(ang)
    circ_y = np.sin(ang)

    # Add circular boundary line
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

    # Configure layout of 2D beam plot
    fig.update_layout(
        title="Beam Pattern",
        xaxis=dict(title="v", range=[-1, 1], showgrid=False, zeroline=False),
        yaxis=dict(title="u", range=[-1, 1], scaleanchor="x", showgrid=False, zeroline=False),
        margin=dict(l=60, r=40, t=50, b=50),
        plot_bgcolor="white"
    )

    # Return fully constructed beam figure
    return fig