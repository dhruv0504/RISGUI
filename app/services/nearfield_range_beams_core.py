# app/services/nearfield_range_beams_core.py
"""
Near-field computation core moved from your app.py.
Provides compute_nearfield_pattern(params) which returns:
  {"figure": plotly Figure, "codebook": list, "status": str}
"""

import io
import numpy as np
import plotly.graph_objects as go

# Fixed RIS size
ANT_X = 32
ANT_Y = 32

def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def quantize_1bit(phi):
    q = np.zeros_like(phi)
    q[(phi > (np.pi/2)) | (phi < (-np.pi/2))] = np.pi
    return q

def precompute_base(
    r_mn_x, r_mn_y, rf_mn, f_pat, elem_pat,
    z_focus, x_pos, y_pos, k0, qe,
):
    Xs, Ys = np.meshgrid(x_pos, y_pos, indexing="ij")  # (Nx,Ny)

    rx = r_mn_x[:, :, None, None].astype(np.float32)
    ry = r_mn_y[:, :, None, None].astype(np.float32)

    Xs4 = Xs[None, None, :, :].astype(np.float32)
    Ys4 = Ys[None, None, :, :].astype(np.float32)

    zf = np.float32(z_focus)

    r_out_plane = np.sqrt((rx - Xs4) ** 2 + (ry - Ys4) ** 2 + zf ** 2).astype(np.float32)
    r_dist = (rf_mn[:, :, None, None].astype(np.float32) + r_out_plane).astype(np.float32)

    theta_out = np.arccos(np.clip(zf / r_out_plane, -1.0, 1.0)).astype(np.float32)
    elem_rad = (np.cos(theta_out) ** qe).astype(np.float32)

    a_mn = (
        f_pat[:, :, None, None].astype(np.float32)
        * elem_pat[:, :, None, None].astype(np.float32)
        * elem_rad
        / (rf_mn[:, :, None, None].astype(np.float32) * r_out_plane)
    ).astype(np.float32)

    base = a_mn * np.exp(-1j * np.float32(k0) * r_dist).astype(np.complex64)  # (32,32,Nx,Ny)
    return base.astype(np.complex64)


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
    lambda0 = 300.0 / float(freq_ghz)
    k0 = (2.0 * np.pi) / lambda0

    rf = np.sqrt(xf**2 + yf**2 + zf**2)

    # Randomization phase
    if randomize and phi_ele_rand is not None:
        phi_ele = np.array(phi_ele_rand, dtype=np.float32)
    else:
        phi_ele = 0.0

    # Element coordinates
    m_idx = np.arange(1, ANT_X + 1)[:, None]
    n_idx = np.arange(1, ANT_Y + 1)[None, :]
    r_mn_x = ((ANT_X - (2 * m_idx - 1)) * ant_size_x) / 2.0
    r_mn_y = ((ANT_Y - (2 * n_idx - 1)) * ant_size_y) / 2.0
    r_mn_x = r_mn_x.astype(np.float32)
    r_mn_y = r_mn_y.astype(np.float32)

    # Feed -> element
    rf_mn_x = r_mn_x - np.float32(xf)
    rf_mn_y = r_mn_y - np.float32(yf)
    rf_mn_z = -np.float32(zf)
    rf_mn = np.sqrt(rf_mn_x**2 + rf_mn_y**2 + rf_mn_z**2).astype(np.float32)

    rf_mn_dot = (-rf_mn_x * np.float32(xf)) + (-rf_mn_y * np.float32(yf)) + (-rf_mn_z * np.float32(zf))
    # Protect division by zero in dot-product normalization
    denom = np.float32(rf) * rf_mn
    denom = np.where(denom == 0, 1e-12, denom)
    theta_f_mn = np.arccos(np.clip(rf_mn_dot / denom, -1.0, 1.0)).astype(np.float32)
    theta_emn = np.arccos(np.clip(np.float32(z_focus) / np.where(rf_mn==0,1e-12,rf_mn), -1.0, 1.0)).astype(np.float32)

    elem_pat = (np.cos(theta_emn) ** qe).astype(np.float32)
    f_pat = (np.cos(theta_f_mn) ** qf).astype(np.float32)

    # Scan grid
    x_pos = np.arange(-x_range, x_range + x_step, x_step, dtype=np.float32)
    y_pos = np.arange(-y_range, y_range + y_step, y_step, dtype=np.float32)

    # Precompute base term once
    base_mn_xy = precompute_base(
        r_mn_x, r_mn_y, rf_mn, f_pat, elem_pat,
        z_focus=np.float32(z_focus),
        x_pos=x_pos, y_pos=y_pos,
        k0=np.float32(k0), qe=qe
    )  # (32,32,Nx,Ny) complex64

    codebook_bits_list = []
    rr_norm_list = []

    # Loop focus points
    for x_focus in x_focus_vals:
        for y_focus in y_focus_vals:
            r_out_mn = np.sqrt(
                (r_mn_x - np.float32(x_focus)) ** 2 +
                (r_mn_y - np.float32(y_focus)) ** 2 +
                np.float32(z_focus) ** 2
            ).astype(np.float32)

            phi_ph_shift = wrap_to_pi((2*np.pi) - np.float32(k0) * (rf_mn + r_out_mn) + phi_ele)
            phi_q = quantize_1bit(phi_ph_shift).astype(np.float32)

            bits = (phi_q / np.pi).astype(np.int8)
            codebook_bits_list.append(bits)

            w = np.exp(1j * phi_q).astype(np.complex64)  # (32,32)
            E_xy = np.einsum("mn,mnxy->xy", w, base_mn_xy)  # (Nx,Ny) complex64

            # safe dB calc
            rr = 20.0 * np.log10(np.abs(E_xy) + 1e-12)
            rr_norm = rr - np.max(rr)
            rr_norm_list.append(rr_norm.astype(np.float32))

    codebook_bits = np.stack(codebook_bits_list, axis=0)  # (N,32,32)
    rr_norm_all = np.stack(rr_norm_list, axis=0)          # (N,Nx,Ny)
    return codebook_bits, rr_norm_all, x_pos, y_pos


def load_phi_ele_rand():
    # Placeholder for user-supplied random matrix (32x32)
    return None


def build_codebook_text(codebook_bits_32x32: np.ndarray) -> str:
    out = io.StringIO()
    for k in range(codebook_bits_32x32.shape[0]):
        bits32 = codebook_bits_32x32[k]              # (32,32)
        bits32x64 = np.repeat(bits32, 2, axis=1)    # (32,64)
        out.write(f"//Mask {k+1}: \n")
        for r in range(bits32x64.shape[0]):
            row = bits32x64[r]
            out.write("{" + ",".join(str(int(v)) for v in row) + "},\n")
        out.write("\n\n")
    return out.getvalue()


def linspace_inclusive(start, step, stop):
    start = float(start)
    step = float(step)
    stop = float(stop)
    if step == 0:
        return np.array([start], dtype=float)
    n = int(np.floor((stop - start) / step + 1 + 1e-9))
    if n <= 0:
        return np.array([], dtype=float)
    vals = start + step * np.arange(n)
    if step > 0:
        vals = vals[vals <= stop + 1e-9]
    else:
        vals = vals[vals >= stop - 1e-9]
    return vals.astype(float)


# Public service callable by Dash wrapper
def compute_nearfield_pattern(params: dict):
    """
    params keys (strings): xi, yi, zi, zr, rand ('On'|'Off'),
      x_start, x_step, x_stop, y_start, y_step, y_stop,
      x_range, y_range, x_scan_step, y_scan_step,
      dr_min, dr_max
    Returns dict: {"figure": fig, "codebook": stored, "status": status}
    """
    # read inputs with defaults
    xi = float(params.get("xi", 0.0))
    yi = float(params.get("yi", 145.0))
    zi = float(params.get("zi", 250.0))
    zr = float(params.get("zr", 300.0))
    rand_value = params.get("rand", "On")

    x_start = float(params.get("x_start", 0.0))
    x_step = float(params.get("x_step", 10.0))
    x_stop = float(params.get("x_stop", 0.0))

    y_start = float(params.get("y_start", 0.0))
    y_step = float(params.get("y_step", 10.0))
    y_stop = float(params.get("y_stop", 0.0))

    x_range = int(params.get("x_range", 1000))
    y_range = int(params.get("y_range", 1000))
    x_scan_step = int(params.get("x_scan_step", 50))
    y_scan_step = int(params.get("y_scan_step", 50))

    dr_min = float(params.get("dr_min", -30.0))
    dr_max = float(params.get("dr_max", 0.0))

    x_focus_vals = linspace_inclusive(x_start, x_step, x_stop)
    y_focus_vals = linspace_inclusive(y_start, y_step, y_stop)
    if x_focus_vals.size == 0 or y_focus_vals.size == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Beam pattern", xaxis_title="X (mm)", yaxis_title="Y (mm)", template="plotly_white")
        return {"figure": empty_fig, "codebook": None, "status": "Focus sweep is empty (check Start/Step/Stop)."}

    randomize = (rand_value == "On")
    phi_ele_rand = load_phi_ele_rand() if randomize else None

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

    rr_maxproj = np.max(rr_norm_all, axis=0)
    rr_disp = np.clip(rr_maxproj, float(dr_min), float(dr_max))

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
    fig.update_layout(
        title=f"Overlapped Radiation Plot (Quantized, Max Projection) | RIS {ANT_X}×{ANT_Y}",
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        template="plotly_white",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    status = f"Generated {codebook_bits.shape[0]} mask(s). Scan grid: {len(x_pos)}×{len(y_pos)}. RIS fixed: {ANT_X}×{ANT_Y}."
    stored = codebook_bits.astype(int).tolist()
    # return 'z' key to satisfy callers/tests that expect a z-grid/result
    return {"figure": fig, "codebook": stored, "status": status, "z": rr_disp}


# Add a small pickle-backed cache wrapper for dict-params functions
import pickle, functools

def _cache_by_params(func, maxsize=128):
    cache = functools.lru_cache(maxsize=maxsize)(lambda key: func(pickle.loads(key)))
    def wrapper(params):
        try:
            key = pickle.dumps(params, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            key = pickle.dumps(str(params), protocol=pickle.HIGHEST_PROTOCOL)
        return cache(key)
    wrapper.cache_clear = cache.cache_clear
    return wrapper

# Wrap compute_nearfield_pattern with caching
compute_nearfield_pattern = _cache_by_params(compute_nearfield_pattern, maxsize=128)
