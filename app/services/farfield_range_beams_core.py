# app/services/farfield_range_beams_core.py

"""
Far-field computation core.
API shape matches your nearfield example: compute_farfield_pattern(params)
returns {"figure": plotly Figure, "codebook": list-or-None, "status": str}

This file intentionally mirrors the structure / style of your nearfield core.
Replace compute_farfield_fast() internals with the full MATLAB -> Python port when ready.
"""

# Used for in-memory string building (codebook text export)
import io

# Numerical computing library
import numpy as np

# Plotly for heatmap visualization
import plotly.graph_objects as go

# LRU cache decorator for performance optimization
from functools import lru_cache


# Fixed RIS size (kept like the MATLAB UI)
ANT_X = 32  # Number of RIS elements in X direction
ANT_Y = 32  # Number of RIS elements in Y direction


# Wrap angle(s) into [-π, π] range
def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# Inclusive linspace helper (MATLAB-like sweep behavior)
def linspace_inclusive(start, step, stop):

    # Convert inputs to float
    start = float(start)
    step = float(step)
    stop = float(stop)

    # If step is zero, return single value
    if step == 0:
        return np.array([start], dtype=float)

    # Compute number of samples
    n = int(np.floor((stop - start) / step + 1 + 1e-9))

    # If invalid sweep range, return empty array
    if n <= 0:
        return np.array([], dtype=float)

    # Generate values
    vals = start + step * np.arange(n)

    # Clip depending on sweep direction
    if step > 0:
        vals = vals[vals <= stop + 1e-9]
    else:
        vals = vals[vals >= stop - 1e-9]

    return vals.astype(float)


# Convert mask bits to downloadable C-style text
def build_codebook_text(codebook_bits_32x32: np.ndarray) -> str:

    """
    Minimal text exporter for codebook bits in the same spirit as your MATLAB writer.
    Input: (N,32,32) integer bits array.
    """

    out = io.StringIO()

    # Loop through masks
    for k in range(codebook_bits_32x32.shape[0]):

        bits32 = codebook_bits_32x32[k]              # (32,32)

        # Expand to 64 columns (replication like nearfield)
        bits32x64 = np.repeat(bits32, 2, axis=1)

        out.write(f"//Mask {k+1}: \n")

        for r in range(bits32x64.shape[0]):
            row = bits32x64[r]
            out.write("{" + ",".join(str(int(v)) for v in row) + "},\n")

        out.write("\n\n")

    return out.getvalue()


# === Placeholder fast far-field engine ===
def compute_farfield_fast(
    pr_start, pr_step, pr_stop, tr_start, tr_step, tr_stop,
    RS1=ANT_X, RS2=ANT_Y,
    ti=0.0, pi_angle=0.0,
    dr_min=-30.0, dr_max=0.0,
    use_randomization=False,
):

    """
    Synthetic far-field generator.
    Replace with real MATLAB port later.
    """

    print("\n===== FARFIELD ENGINE START =====")
    print("Reflection Phi Sweep:", pr_start, pr_step, pr_stop)
    print("Reflection Theta Sweep:", tr_start, tr_step, tr_stop)
    print("RIS Size:", RS1, "x", RS2)
    print("Dynamic Range:", dr_min, "to", dr_max)
    print("Randomization Enabled:", use_randomization)

    # Create observation grid
    x = np.linspace(-1.0, 1.0, 101)
    y = np.linspace(-1.0, 1.0, 101)
    X, Y = np.meshgrid(x, y)

    print("Observation Grid Shape:", X.shape)

    # Generate sweep values
    phi_vals = linspace_inclusive(pr_start, pr_step, pr_stop)
    theta_vals = linspace_inclusive(tr_start, tr_step, tr_stop)

    print("Phi Sweep Count:", phi_vals.size)
    print("Theta Sweep Count:", theta_vals.size)

    if phi_vals.size == 0 or theta_vals.size == 0:

        print("⚠ Empty sweep detected")

        Z = np.zeros_like(X)

        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=Z, x=x, y=y, zmin=dr_min, zmax=dr_max,
                                 colorbar=dict(title="Magnitude (dB)")))

        fig.update_layout(title="Far-field (empty sweep)",
                          xaxis_title="u",
                          yaxis_title="v",
                          template="plotly_white")

        return {"figure": fig,
                "codebook": None,
                "status": "Empty reflection sweep (check Start/Step/Stop)."}

    masks_bits = []
    rr_norm_list = []

    # Generate synthetic beam patterns
    for p in phi_vals:
        for t in theta_vals:

            print("Generating synthetic beam at (phi,theta):", p, t)

            cx = 0.5 * np.cos(np.deg2rad(p)) * np.cos(np.deg2rad(t))
            cy = 0.5 * np.sin(np.deg2rad(p)) * np.cos(np.deg2rad(t))

            sigma = 0.25

            Z = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))

            rr = 20.0 * np.log10(np.abs(Z) + 1e-12)
            rr_norm = rr - np.max(rr)

            print("Beam Max dB:", np.max(rr_norm))
            print("Beam Min dB:", np.min(rr_norm))

            rr_norm_list.append(rr_norm.astype(np.float32))

            # Synthetic 1-bit mask
            bits = (np.random.RandomState(
                int((p + t) * 100) % 2**31).randint(0,2,(RS1,RS2))
            ).astype(int)

            masks_bits.append(bits)

    rr_stack = np.stack(rr_norm_list, axis=0)

    rr_maxproj = np.max(rr_stack, axis=0)

    rr_disp = np.clip(rr_maxproj, dr_min, dr_max)

    print("Final Max Projection dB:", np.max(rr_disp))
    print("Final Min Projection dB:", np.min(rr_disp))

    fig = go.Figure(
        data=go.Heatmap(
            z=rr_disp,
            x=x,
            y=y,
            colorbar=dict(title="Magnitude (dB)"),
            zmin=float(dr_min),
            zmax=float(dr_max),
        )
    )

    fig.update_layout(
        title=f"Far-field Overlapped Radiation (synthetic) | RIS {RS1}×{RS2}",
        xaxis_title="u",
        yaxis_title="v",
        template="plotly_white",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    codebook_bits = np.stack(masks_bits, axis=0).astype(int)

    print("Total Masks Generated:", codebook_bits.shape[0])

    status = f"Generated {codebook_bits.shape[0]} synthetic mask(s)."

    print("===== FARFIELD ENGINE END =====\n")

    return {"figure": fig,
            "codebook": codebook_bits,
            "status": status}


# Cache for performance
compute_farfield_fast = lru_cache(maxsize=128)(compute_farfield_fast)


# === Public API used by Dash wrapper ===
def compute_farfield_pattern(params: dict):

    print("\n===== PUBLIC API CALL =====")

    # Parse parameters
    ti = float(params.get("ti", 0.0))
    pi_angle = float(params.get("pi", 0.0))

    prstart = float(params.get("prstart", 0.0))
    prstep = float(params.get("prstep", 1.0))
    prstop  = float(params.get("prstop", 0.0))

    trstart = float(params.get("trstart", 0.0))
    trstep  = float(params.get("trstep", 1.0))
    trstop  = float(params.get("trstop", 0.0))

    rand_value = params.get("rand", "Off")
    randomize = (rand_value == "On")

    rs1 = int(params.get("rs1", ANT_X))
    rs2 = int(params.get("rs2", ANT_Y))

    dr_min = float(params.get("dr_min", -30.0))
    dr_max = float(params.get("dr_max", 0.0))

    print("Incident Angles:", ti, pi_angle)
    print("Reflection Sweep:", prstart, prstep, prstop)
    print("Theta Sweep:", trstart, trstep, trstop)

    result = compute_farfield_fast(
        prstart, prstep, prstop,
        trstart, trstep, trstop,
        RS1=rs1, RS2=rs2,
        ti=ti, pi_angle=pi_angle,
        dr_min=dr_min, dr_max=dr_max,
        use_randomization=randomize,
    )

    codebook = result.get("codebook")
    stored = None

    if codebook is not None:
        stored = codebook.astype(int).tolist()

    print("Status:", result.get("status", ""))
    print("===== PUBLIC API END =====\n")

    return {"figure": result.get("figure"),
            "codebook": stored,
            "status": result.get("status", "")}