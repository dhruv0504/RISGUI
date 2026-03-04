# app/services/nearfield_range_beams_core.py
# Near-field range beam computation module

"""
Near-field computation core moved from your app.py.
Provides compute_nearfield_pattern(params) which returns:
  {"figure": plotly Figure, "codebook": list, "status": str}
"""

# Used to build string buffer for exporting codebook text
import io

# Numerical computing library for array math
import numpy as np

# Plotly for interactive heatmap visualization
import plotly.graph_objects as go


# Fixed RIS size (32 x 32 elements)
ANT_X = 32
ANT_Y = 32


# Wrap angle into [-π, π] range
def wrap_to_pi(x):
    # Ensures phase continuity
    return (x + np.pi) % (2 * np.pi) - np.pi


# Quantize phase to 1-bit (0 or π)
def quantize_1bit(phi):

    # Initialize output array same shape as phi
    q = np.zeros_like(phi)

    # Set π where phase exceeds ±π/2
    q[(phi > (np.pi/2)) | (phi < (-np.pi/2))] = np.pi

    return q


# Precompute base term for fast near-field evaluation
def precompute_base(
    r_mn_x, r_mn_y, rf_mn, f_pat, elem_pat,
    z_focus, x_pos, y_pos, k0, qe,
):

    print("\n==== PRECOMPUTE BASE START ====")

    # Build scan grid (X,Y)
    Xs, Ys = np.meshgrid(x_pos, y_pos, indexing="ij")  # (Nx,Ny)

    print("Scan grid shape:", Xs.shape)

    # Expand element positions into 4D tensors for broadcasting
    rx = r_mn_x[:, :, None, None].astype(np.float32)
    ry = r_mn_y[:, :, None, None].astype(np.float32)

    # Expand scan grid to 4D for vectorized computation
    Xs4 = Xs[None, None, :, :].astype(np.float32)
    Ys4 = Ys[None, None, :, :].astype(np.float32)

    # Convert focus Z to float32
    zf = np.float32(z_focus)

    # Compute distance from each element to each scan point
    r_out_plane = np.sqrt(
        (rx - Xs4) ** 2 +
        (ry - Ys4) ** 2 +
        zf ** 2
    ).astype(np.float32)

    # Total path length = feed-element + element-scan
    r_dist = (
        rf_mn[:, :, None, None].astype(np.float32)
        + r_out_plane
    ).astype(np.float32)

    # Compute output angle
    theta_out = np.arccos(
        np.clip(zf / r_out_plane, -1.0, 1.0)
    ).astype(np.float32)

    # Element radiation pattern
    elem_rad = (np.cos(theta_out) ** qe).astype(np.float32)

    # Compute amplitude weighting
    a_mn = (
        f_pat[:, :, None, None].astype(np.float32)
        * elem_pat[:, :, None, None].astype(np.float32)
        * elem_rad
        / (rf_mn[:, :, None, None].astype(np.float32) * r_out_plane)
    ).astype(np.float32)

    print("Amplitude tensor shape:", a_mn.shape)

    # Compute base complex exponential term
    base = a_mn * np.exp(
        -1j * np.float32(k0) * r_dist
    ).astype(np.complex64)

    print("Base tensor shape:", base.shape)
    print("==== PRECOMPUTE BASE END ====\n")

    return base.astype(np.complex64)

# Fast near-field beam generator (vectorized implementation)
def generate_nearfield_fast(
    ANT_X=32, ANT_Y=32,
    ant_size_x=5.4, ant_size_y=5.4,
    freq_ghz=27.2, qe=1, qf=18,
    z_focus=300.0,
    xf=0.0, yf=145.0, zf=250.0,
    x_focus_vals=(0.0,), y_focus_vals=(0.0,),
    randomize=False, phi_ele_rand=None,
    x_range=1000, y_range=1000,
    x_step=50, y_step=50,
):

    print("\n========== GENERATE NEARFIELD FAST START ==========")

    # Compute wavelength in mm
    lambda0 = 300.0 / float(freq_ghz)

    # Compute wave number
    k0 = (2.0 * np.pi) / lambda0

    print("Frequency (GHz):", freq_ghz)
    print("Wavelength (mm):", lambda0)
    print("Wavenumber k0:", k0)

    # Compute feed distance from origin
    rf = np.sqrt(xf**2 + yf**2 + zf**2)

    print("Feed position:", (xf, yf, zf))
    print("Feed distance:", rf)

    # Randomization phase handling
    if randomize and phi_ele_rand is not None:

        # Use provided random phase matrix
        phi_ele = np.array(phi_ele_rand, dtype=np.float32)

        print("Random phase enabled.")

    else:

        # No random phase
        phi_ele = 0.0

        print("Random phase disabled.")

    # Element indices for X and Y directions
    m_idx = np.arange(1, ANT_X + 1)[:, None]
    n_idx = np.arange(1, ANT_Y + 1)[None, :]

    # Compute element X coordinates
    r_mn_x = ((ANT_X - (2 * m_idx - 1)) * ant_size_x) / 2.0

    # Compute element Y coordinates
    r_mn_y = ((ANT_Y - (2 * n_idx - 1)) * ant_size_y) / 2.0

    # Convert to float32
    r_mn_x = r_mn_x.astype(np.float32)
    r_mn_y = r_mn_y.astype(np.float32)

    print("Element grid shape:", r_mn_x.shape)

    # Compute feed-to-element distances
    rf_mn_x = r_mn_x - np.float32(xf)
    rf_mn_y = r_mn_y - np.float32(yf)
    rf_mn_z = -np.float32(zf)

    rf_mn = np.sqrt(
        rf_mn_x**2 +
        rf_mn_y**2 +
        rf_mn_z**2
    ).astype(np.float32)

    print("Min feed-element distance:", np.min(rf_mn))
    print("Max feed-element distance:", np.max(rf_mn))

    # Compute dot-product for feed pattern angle
    rf_mn_dot = (
        (-rf_mn_x * np.float32(xf)) +
        (-rf_mn_y * np.float32(yf)) +
        (-rf_mn_z * np.float32(zf))
    )

    # Avoid division by zero
    denom = np.float32(rf) * rf_mn
    denom = np.where(denom == 0, 1e-12, denom)

    # Compute feed-element angle
    theta_f_mn = np.arccos(
        np.clip(rf_mn_dot / denom, -1.0, 1.0)
    ).astype(np.float32)

    # Compute element radiation angle
    theta_emn = np.arccos(
        np.clip(
            np.float32(z_focus) /
            np.where(rf_mn == 0, 1e-12, rf_mn),
            -1.0, 1.0
        )
    ).astype(np.float32)

    # Element pattern
    elem_pat = (np.cos(theta_emn) ** qe).astype(np.float32)

    # Feed pattern
    f_pat = (np.cos(theta_f_mn) ** qf).astype(np.float32)

    print("Element pattern range:",
          np.min(elem_pat), np.max(elem_pat))
    print("Feed pattern range:",
          np.min(f_pat), np.max(f_pat))

    # Build scan grid
    x_pos = np.arange(-x_range, x_range + x_step, x_step, dtype=np.float32)
    y_pos = np.arange(-y_range, y_range + y_step, y_step, dtype=np.float32)

    print("Scan grid size:", len(x_pos), "x", len(y_pos))

    # Precompute base term (heavy tensor)
    base_mn_xy = precompute_base(
        r_mn_x, r_mn_y, rf_mn, f_pat, elem_pat,
        z_focus=np.float32(z_focus),
        x_pos=x_pos, y_pos=y_pos,
        k0=np.float32(k0), qe=qe
    )

    # Initialize storage lists
    codebook_bits_list = []
    rr_norm_list = []

    # Loop over focus sweep values
    for x_focus in x_focus_vals:
        for y_focus in y_focus_vals:

            print("Computing focus at:", x_focus, y_focus)

            # Compute element-to-focus distance
            r_out_mn = np.sqrt(
                (r_mn_x - np.float32(x_focus)) ** 2 +
                (r_mn_y - np.float32(y_focus)) ** 2 +
                np.float32(z_focus) ** 2
            ).astype(np.float32)

            # Compute phase shift
            phi_ph_shift = wrap_to_pi(
                (2*np.pi)
                - np.float32(k0) * (rf_mn + r_out_mn)
                + phi_ele
            )

            # Quantize phase
            phi_q = quantize_1bit(phi_ph_shift).astype(np.float32)

            # Convert to binary bits
            bits = (phi_q / np.pi).astype(np.int8)

            print("Mask unique values:", np.unique(bits))

            # Store mask
            codebook_bits_list.append(bits)

            # Compute element weighting
            w = np.exp(1j * phi_q).astype(np.complex64)

            # Compute near-field via Einstein summation
            E_xy = np.einsum("mn,mnxy->xy", w, base_mn_xy)

            # Convert to dB safely
            rr = 20.0 * np.log10(np.abs(E_xy) + 1e-12)

            # Normalize
            rr_norm = rr - np.max(rr)

            print("Beam max dB:", np.max(rr_norm))
            print("Beam min dB:", np.min(rr_norm))

            rr_norm_list.append(rr_norm.astype(np.float32))

    # Stack masks
    codebook_bits = np.stack(codebook_bits_list, axis=0)

    # Stack radiation patterns
    rr_norm_all = np.stack(rr_norm_list, axis=0)

    print("Total masks generated:", codebook_bits.shape[0])
    print("========== GENERATE NEARFIELD FAST END ==========\n")

    return codebook_bits, rr_norm_all, x_pos, y_pos

# Load optional random phase matrix (placeholder for future extension)
def load_phi_ele_rand():

    # Print information that this is placeholder
    print("load_phi_ele_rand() called - returning None (placeholder)")

    # Currently no user-provided random matrix
    return None


# Convert binary codebook masks into text format
def build_codebook_text(codebook_bits_32x32: np.ndarray) -> str:

    # Print start banner
    print("\n==== BUILD CODEBOOK TEXT START ====")

    # Create in-memory string buffer
    out = io.StringIO()

    # Loop over each mask
    for k in range(codebook_bits_32x32.shape[0]):

        # Extract 32x32 mask
        bits32 = codebook_bits_32x32[k]              # (32,32)

        print(f"Processing mask {k+1}, shape:", bits32.shape)

        # Expand to 32x64 by repeating columns (MATLAB-compatible format)
        bits32x64 = np.repeat(bits32, 2, axis=1)    # (32,64)

        # Write mask header
        out.write(f"//Mask {k+1}: \n")

        # Loop through rows
        for r in range(bits32x64.shape[0]):

            # Extract row
            row = bits32x64[r]

            # Convert to comma-separated string and write
            out.write("{" + ",".join(str(int(v)) for v in row) + "},\n")

        # Separate masks
        out.write("\n\n")

    print("Total masks exported:", codebook_bits_32x32.shape[0])
    print("==== BUILD CODEBOOK TEXT END ====\n")

    # Return entire text content
    return out.getvalue()


# Inclusive linspace helper (MATLAB-like sweep)
def linspace_inclusive(start, step, stop):

    # Convert inputs to float
    start = float(start)
    step = float(step)
    stop = float(stop)

    print("linspace_inclusive inputs:", start, step, stop)

    # If step is zero, return single value
    if step == 0:
        print("Step is zero. Returning single value.")
        return np.array([start], dtype=float)

    # Compute number of steps
    n = int(np.floor((stop - start) / step + 1 + 1e-9))

    # If no valid points, return empty array
    if n <= 0:
        print("No valid sweep points.")
        return np.array([], dtype=float)

    # Generate values
    vals = start + step * np.arange(n)

    # Apply clipping depending on direction
    if step > 0:
        vals = vals[vals <= stop + 1e-9]
    else:
        vals = vals[vals >= stop - 1e-9]

    print("Generated sweep length:", len(vals))

    return vals.astype(float)

# Public service callable by Dash wrapper
# Public service callable by Dash wrapper
def compute_nearfield_pattern(params: dict):

    """
    params keys (strings): xi, yi, zi, zr, rand ('On'|'Off'),
      x_start, x_step, x_stop, y_start, y_step, y_stop,
      x_range, y_range, x_scan_step, y_scan_step,
      dr_min, dr_max
    Returns dict: {"figure": fig, "codebook": stored, "status": status}
    """

    print("\n========== COMPUTE NEARFIELD PATTERN START ==========")

    # Read transmitter (feed) coordinates
    xi = float(params.get("xi", 0.0))
    yi = float(params.get("yi", 145.0))
    zi = float(params.get("zi", 250.0))

    # Read focus Z position
    zr = float(params.get("zr", 300.0))

    # Randomization flag
    rand_value = params.get("rand", "On")

    print("Feed position:", (xi, yi, zi))
    print("Focus Z:", zr)
    print("Randomization mode:", rand_value)

    # Read focus sweep parameters (X direction)
    x_start = float(params.get("x_start", 0.0))
    x_step = float(params.get("x_step", 10.0))
    x_stop = float(params.get("x_stop", 0.0))

    # Read focus sweep parameters (Y direction)
    y_start = float(params.get("y_start", 0.0))
    y_step = float(params.get("y_step", 10.0))
    y_stop = float(params.get("y_stop", 0.0))

    # Scan range settings
    x_range = int(params.get("x_range", 1000))
    y_range = int(params.get("y_range", 1000))
    x_scan_step = int(params.get("x_scan_step", 50))
    y_scan_step = int(params.get("y_scan_step", 50))

    # Dynamic range
    dr_min = float(params.get("dr_min", -30.0))
    dr_max = float(params.get("dr_max", 0.0))

    print("Focus X sweep:", x_start, x_step, x_stop)
    print("Focus Y sweep:", y_start, y_step, y_stop)
    print("Scan range:", x_range, y_range)
    print("Dynamic range:", dr_min, dr_max)

    # Generate inclusive sweep arrays
    x_focus_vals = linspace_inclusive(x_start, x_step, x_stop)
    y_focus_vals = linspace_inclusive(y_start, y_step, y_stop)

    # Check if sweep is empty
    if x_focus_vals.size == 0 or y_focus_vals.size == 0:

        print("Focus sweep empty. Returning empty figure.")

        empty_fig = go.Figure()

        empty_fig.update_layout(
            title="Beam pattern",
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            template="plotly_white"
        )

        return {
            "figure": empty_fig,
            "codebook": None,
            "status": "Focus sweep is empty (check Start/Step/Stop)."
        }

    # Determine if randomization is enabled
    randomize = (rand_value == "On")

    # Load random phase matrix if needed
    phi_ele_rand = load_phi_ele_rand() if randomize else None

    # Call fast nearfield generator
    codebook_bits, rr_norm_all, x_pos, y_pos = generate_nearfield_fast(
        z_focus=float(zr),
        xf=float(xi), yf=float(yi), zf=float(zi),
        x_focus_vals=x_focus_vals,
        y_focus_vals=y_focus_vals,
        randomize=randomize,
        phi_ele_rand=phi_ele_rand,
        x_range=int(x_range), y_range=int(y_range),
        x_step=int(x_scan_step), y_step=int(y_scan_step),
    )

    print("Generated masks:", codebook_bits.shape[0])
    print("Scan grid size:", len(x_pos), "x", len(y_pos))

    # Compute maximum projection across masks
    rr_maxproj = np.max(rr_norm_all, axis=0)

    # Apply clipping
    rr_disp = np.clip(rr_maxproj, float(dr_min), float(dr_max))

    print("Final projection max dB:", np.max(rr_disp))
    print("Final projection min dB:", np.min(rr_disp))

    # Build Plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=rr_disp.T,
            x=x_pos,
            y=y_pos,
            colorbar=dict(title="Magnitude (dB)"),
            zmin=float(dr_min),
            zmax=float(dr_max),
        )
    )

    # Configure layout
    fig.update_layout(
        title=f"Overlapped Radiation Plot (Quantized, Max Projection) | RIS {ANT_X}×{ANT_Y}",
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        template="plotly_white",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    # Build status message
    status = (
        f"Generated {codebook_bits.shape[0]} mask(s). "
        f"Scan grid: {len(x_pos)}×{len(y_pos)}. "
        f"RIS fixed: {ANT_X}×{ANT_Y}."
    )

    # Convert mask array to list for storage
    stored = codebook_bits.astype(int).tolist()

    print("========== COMPUTE NEARFIELD PATTERN END ==========\n")

    # Return results
    return {
        "figure": fig,
        "codebook": stored,
        "status": status,
        "z": rr_disp
    }


# Add pickle-backed cache wrapper
import pickle, functools


def _cache_by_params(func, maxsize=128):

    # Create LRU cache using serialized params
    cache = functools.lru_cache(maxsize=maxsize)(
        lambda key: func(pickle.loads(key))
    )

    # Wrapper function
    def wrapper(params):

        # Serialize params safely
        try:
            key = pickle.dumps(params, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            key = pickle.dumps(str(params), protocol=pickle.HIGHEST_PROTOCOL)

        # Call cached function
        return cache(key)

    # Expose cache clear method
    wrapper.cache_clear = cache.cache_clear

    return wrapper


# Wrap compute_nearfield_pattern with caching
compute_nearfield_pattern = _cache_by_params(
    compute_nearfield_pattern,
    maxsize=128
)