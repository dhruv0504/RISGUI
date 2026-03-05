# app/services/nearfield_core.py
# This file implements the near-field electromagnetic computation service.

"""
Near-field compute & plotting service based on your MATLAB implementation.
RIS size is fixed to 32x32. Exposes compute_nearfield(...) and helper builders.
"""

# Used to capture full exception stack trace when an error occurs
import traceback

# Numerical computing library for arrays, vectorized math, linear algebra
import numpy as np

# Plotly graph objects used for building interactive 2D/3D visualizations
import plotly.graph_objects as go

# PIL library for loading texture images (used in RIS plane visualization)
from PIL import Image

# Used for checking if texture image exists on disk
import os

# LRU cache decorator to speed repeated identical nearfield computations
from functools import lru_cache


# constants / default random sample (from your provided vector)

# Predefined random per-element phase sample (radians)
# Used when randomize=True to simulate element randomness
phi_ele_rand_sample = np.array([
 0.219827130698740, 2.98461967174048, 0.497030697989493, 0.899896229536556,
 2.15867977892526, 0.443435248995979, 1.60876765916016, 2.26611537562385,
 2.91805305438595, 2.29997207687589, 2.35571703325974, 1.27963227277936,
 0.752385351454657, 1.63630002613963, 0.688251825085977, 2.64643965036121
])

# fixed RIS

# Number of RIS elements in X direction (columns)
STATIC_RS1 = 32

# Number of RIS elements in Y direction (rows)
STATIC_RS2 = 32


# Wrap any angle into [-π, π] range
def wrap_to_pi(x):
    # Add π, apply modulo 2π, subtract π
    # Ensures phase wrapping consistency
    return (x + np.pi) % (2*np.pi) - np.pi

# Main near-field computation function
def compute_nearfield(xi, yi, zi, xr, yr, zr, zcut,
                      dr1=-20, dr2=0, randomize=True,
                      fc_mhz=27.2, ant_size_x=5.4, ant_size_y=5.4,
                      qe=1, qf=18,
                      x_range=1000, y_range=1000, x_step=50, y_step=50,
                      verbose=False, phi_ele_rand_sample_arg=None):

    # Documentation string describing purpose and outputs
    """
    Compute near-field quantities.
    Returns dict with masks, illumination, XY/YZ cuts, element positions + incident/reflected unit vectors.
    """

    try:

        # Print computation start banner
        print("\n========== NEARFIELD COMPUTATION START ==========")

        # Force RIS size fixed to 32x32 (overrides any external sizing)
        ant_x = STATIC_RS1
        ant_y = STATIC_RS2

        # Print RIS dimensions
        print("RIS Size:", ant_x, "x", ant_y)

        # Compute wavelength in millimeters using c ≈ 300 mm/ns
        lambda0 = 300.0 / fc_mhz

        # Compute wavenumber k0 = 2π / λ
        k0 = 2.0 * np.pi / lambda0

        # Print frequency-related parameters
        print("Frequency (MHz):", fc_mhz)
        print("Wavelength (mm):", lambda0)
        print("Wavenumber k0:", k0)

        # Compute centered element X positions using MATLAB indexing style
        x_vals = ((ant_x - (2*np.arange(1, ant_x+1)-1))*ant_size_x)/2.0

        # Compute centered element Y positions
        y_vals = ((ant_y - (2*np.arange(1, ant_y+1)-1))*ant_size_y)/2.0

        # Build 2D meshgrid of element coordinates
        X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals, indexing='xy')

        # Print element spacing
        print("Element spacing X:", ant_size_x)
        print("Element spacing Y:", ant_size_y)
        print("Element grid shape:", X_mesh.shape)

        # Feed (transmitter) coordinates converted to float
        xf, yf, zf = float(xi), float(yi), float(zi)

        # Receiver (focus point) coordinates converted to float
        xr_f, yr_f, zr_f = float(xr), float(yr), float(zr)

        # Compute distance from feed to origin
        rf = np.sqrt(xf*xf + yf*yf + zf*zf)

        # Print feed and receiver positions
        print("Feed position:", (xf, yf, zf))
        print("Receiver position:", (xr_f, yr_f, zr_f))
        print("Feed distance from origin:", rf)

        # Build incident direction vector (pointing from feed toward RIS origin)
        incident_vector = np.array([ -xf, -yf, -zf ], dtype=float)

        # Compute magnitude of incident vector
        incident_norm = np.linalg.norm(incident_vector)

        # Normalize incident vector if valid
        if incident_norm > 0:
            incident_vector = incident_vector / incident_norm
        else:
            # Fallback direction if zero vector
            incident_vector = np.array([0.0, 0.0, 1.0])

        # Build reflected direction vector (from RIS origin to receiver)
        reflected_vector = np.array([ xr_f, yr_f, zr_f ], dtype=float)

        # Compute magnitude of reflected vector
        ref_norm = np.linalg.norm(reflected_vector)

        # Normalize reflected vector if valid
        if ref_norm > 0:
            reflected_vector = reflected_vector / ref_norm
        else:
            # Fallback direction if zero vector
            reflected_vector = np.array([0.0, 0.0, 1.0])

        # Print unit vectors
        print("Incident unit vector:", incident_vector)
        print("Reflected unit vector:", reflected_vector)

        # Compute feed-to-element X offsets
        rf_mn_x = X_mesh - xf

        # Compute feed-to-element Y offsets
        rf_mn_y = Y_mesh - yf

        # Compute feed-to-element Z offsets
        rf_mn_z = - zf

        # Compute Euclidean distance from feed to each RIS element
        rf_mn = np.sqrt(rf_mn_x**2 + rf_mn_y**2 + rf_mn_z**2)

        # Print feed-element distance range
        print("Min feed-element distance:", np.min(rf_mn))
        print("Max feed-element distance:", np.max(rf_mn))

        # Compute dot product components for angle calculation
        rf_mn_x_dot = - rf_mn_x * xf
        rf_mn_y_dot = - rf_mn_y * yf
        rf_mn_z_dot = - rf_mn_z * zf

        # Total dot product
        rf_mn_dot = rf_mn_x_dot + rf_mn_y_dot + rf_mn_z_dot

        # Multiply magnitudes for denominator
        rf_mn_new_mod = rf * rf_mn

        # Compute angle between feed direction and element direction
        with np.errstate(invalid='ignore'):
            theta_f_mn = np.arccos(
                np.clip(rf_mn_dot / (rf_mn_new_mod + 1e-12), -1.0, 1.0)
            )

        # Define focus Z position
        z_focus = zr_f

        # Compute distance from each element to focus point
        r_out_mn = np.sqrt(
            (X_mesh - xr_f)**2 +
            (Y_mesh - yr_f)**2 +
            (z_focus)**2
        )

        # Print element-focus distance range
        print("Min element-focus distance:", np.min(r_out_mn))
        print("Max element-focus distance:", np.max(r_out_mn))

        # Generate per-element random phase if enabled
        if not randomize:

            # If randomization disabled, use zero phase offset
            phi_ele_rand = np.zeros_like(X_mesh)

        else:

            # Total number of RIS elements
            nels = ant_x * ant_y

            # Use provided sample or global
            sample = phi_ele_rand_sample_arg if phi_ele_rand_sample_arg is not None else phi_ele_rand_sample

            # Copy predefined random sample and flatten to 1D
            print("sample type:", type(sample))
            v = np.asarray(sample).copy().ravel()

            # If sample smaller than required, tile (repeat) values
            if v.size < nels:
                v = np.tile(v, int(np.ceil(nels / v.size)))[:nels]
            else:
                v = v[:nels]

            # Reshape back into 2D element grid
            phi_ele_rand = v.reshape((ant_y, ant_x))

        # Compute required phase shift per element
        # Total path phase = feed->element + element->focus
        phi_ph_shift = wrap_to_pi(
            2.0*np.pi - k0 * (rf_mn + r_out_mn) + phi_ele_rand
        )

        # Quantize phase to 1-bit:
        # If phase > π/2 or < -π/2 → π
        # Else → 0
        phi_ph_shift_q = np.where(
            (phi_ph_shift > (np.pi/2.0)) |
            (phi_ph_shift < -(np.pi/2.0)),
            np.pi,
            0.0
        )

        # Convert to binary mask (0 or 1)
        codeword = (phi_ph_shift_q / np.pi).astype(int)

        # Print mask statistics
        print("Mask unique states:", np.unique(codeword))
        print("Number of 0 elements:", np.sum(codeword == 0))
        print("Number of 1 elements:", np.sum(codeword == 1))

        # Compute element pattern relative to center focus
        with np.errstate(invalid='ignore'):
            theta_emn_center = np.arccos(
                np.clip(
                    z_focus /
                    (np.sqrt((X_mesh)**2 + (Y_mesh)**2 + z_focus**2) + 1e-12),
                    -1.0,
                    1.0
                )
            )

        # Element radiation pattern (cos^qe)
        elem_pat = np.power(np.cos(theta_emn_center), qe)

        # Feed radiation pattern (cos^qf)
        f_pat = np.power(np.cos(theta_f_mn), qf)

        # Compute output angle for radiation
        with np.errstate(invalid='ignore'):
            theta_out_mn = np.arccos(
                np.clip(z_focus / (r_out_mn + 1e-12), -1.0, 1.0)
            )

        # Element radiation pattern toward focus
        elem_rad = np.power(np.cos(theta_out_mn), qe)

        # Distance denominator term
        denom = (rf_mn * r_out_mn) + 1e-12

        # Final amplitude weighting per element
        a_mn = (f_pat * elem_pat * elem_rad) / denom

        # Compute illumination magnitude in dB
        im_db = 20.0 * np.log10(np.abs(a_mn) + 1e-12)

        # Find maximum illumination
        im_max = np.nanmax(im_db)

        # Normalize illumination
        im_norm = im_db - im_max

        # Print illumination stats
        print("Illumination max dB:", np.max(im_norm))
        print("Illumination min dB:", np.min(im_norm))

        # === XY CUT COMPUTATION ===

        # Define scanning grid in X direction
        x_scan_vals = np.arange(-x_range, x_range + 1, x_step)

        # Define scanning grid in Y direction
        y_scan_vals = np.arange(-y_range, y_range + 1, y_step)

        # Compute grid sizes
        nx = x_scan_vals.size
        ny = y_scan_vals.size

        print("XY grid size:", nx, "x", ny)

        # Initialize complex field matrix
        E_rad_total = np.zeros((nx, ny), dtype=complex)

        # Loop over RIS elements
        for m in range(ant_x):
            for n in range(ant_y):

                # Element position
                xm = X_mesh[n, m]
                ym = Y_mesh[n, m]

                # Feed distance for this element
                rf_mn_val = rf_mn[n, m]

                # Build scan mesh
                Xg, Yg = np.meshgrid(x_scan_vals, y_scan_vals, indexing='xy')

                # Distance from element to scan plane
                r_out_plane = np.sqrt(
                    (xm - Xg)**2 +
                    (ym - Yg)**2 +
                    (zcut)**2
                )

                # Total path length
                r_dist = rf_mn_val + r_out_plane

                # Compute phase contribution
                elem_phase = (
                    np.exp(-1j * k0 * r_dist) *
                    np.exp(-1j * phi_ph_shift[n, m])
                )

                # Element amplitude
                a_val = a_mn[n, m]

                # Accumulate total field
                E_rad_total += a_val * elem_phase

        # Convert XY field to dB
        rr = 20.0 * np.log10(np.abs(E_rad_total) + 1e-12)

        # Normalize
        rr_max = np.nanmax(rr)
        rr_norm = rr - rr_max

        # Clip to dynamic range
        rr_clip = np.clip(rr_norm, dr1, dr2)

        print("XY Cut max dB:", np.max(rr_norm))
        print("XY Cut min dB:", np.min(rr_norm))


        # === YZ CUT COMPUTATION ===

        # Define scanning grid in Z direction
        z_scan_vals = np.arange(0, 2000 + 1, 50)

        # Define scanning grid in Y direction for YZ plane
        y_scan_vals_yz = np.arange(-2000, 2000 + 1, 50)

        # Compute grid sizes
        nz = z_scan_vals.size
        ny_yz = y_scan_vals_yz.size

        print("YZ grid size:", nz, "x", ny_yz)

        # Initialize complex field matrix for YZ cut
        E_rad_yz = np.zeros((nz, ny_yz), dtype=complex)

        # Define X focus coordinate (fixed during YZ scan)
        x_focus = xr_f

        # Loop over RIS elements
        for m in range(ant_x):
            for n in range(ant_y):

                # Element coordinates
                xm = X_mesh[n, m]
                ym = Y_mesh[n, m]

                # Feed-element distance for this element
                rf_mn_val = rf_mn[n, m]

                # Build YZ scan mesh
                Zg, Yg = np.meshgrid(z_scan_vals, y_scan_vals_yz, indexing='xy')

                # Compute distance from element to each scan point
                r_out_plane = np.sqrt(
                    (xm - x_focus)**2 +
                    (ym - Yg)**2 +
                    (Zg)**2
                )

                # Total path length
                r_dist = rf_mn_val + r_out_plane

                # Compute element phase contribution
                elem_phase = (
                    np.exp(-1j * k0 * r_dist) *
                    np.exp(-1j * phi_ph_shift[n, m])
                )

                # Element amplitude weighting
                a_val = a_mn[n, m]

                # Accumulate field contribution
                E_rad_yz += a_val * elem_phase.T

        # Convert YZ field to dB
        rr_yz = 20.0 * np.log10(np.abs(E_rad_yz) + 1e-12)

        # Normalize
        rr_yz_max = np.nanmax(rr_yz)
        rr_yz_norm = rr_yz - rr_yz_max

        # Clip dynamic range
        rr_yz_clip = np.clip(rr_yz_norm, dr1, dr2)

        print("YZ Cut max dB:", np.max(rr_yz_norm))
        print("YZ Cut min dB:", np.min(rr_yz_norm))

        # === Compute angles in degrees ===

        # Incident elevation angle
        theta_inc = np.arctan2(np.sqrt(xi*xi + yi*yi), zi) if (zi != 0) else 0.0

        # Incident azimuth angle
        phi_inc = np.arctan2(yi, xi) if (xi != 0 or yi != 0) else 0.0

        # Reflected elevation angle
        theta_ref = np.arctan2(
            np.sqrt(xr_f*xr_f + yr_f*yr_f), zr_f
        ) if (zr_f != 0) else 0.0

        # Reflected azimuth angle
        phi_ref = np.arctan2(
            yr_f, xr_f
        ) if (xr_f != 0 or yr_f != 0) else 0.0

        print("Incident angles (deg):",
              np.degrees(theta_inc),
              np.degrees(phi_inc))

        print("Reflected angles (deg):",
              np.degrees(theta_ref),
              np.degrees(phi_ref))

        # Store key parameters in dictionary
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

        # Print end banner
        print("========== NEARFIELD COMPUTATION END ==========\n")

        # Return all computed data (unchanged structure)
        return {
            "mask": codeword,
            "phi_ph_shift_q": phi_ph_shift_q,
            "phi_ph_shift": phi_ph_shift,
            "im_norm": im_norm,
            "rr_norm_xy": rr_norm,
            "rr_clip_xy": rr_clip,
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

    # Catch any runtime errors
    except Exception as e:

        # Capture full traceback
        tb = traceback.format_exc()

        # Print error message
        print("compute_nearfield error:", e)

        # Print full traceback
        print(tb)

        # Return error dictionary
        return {"error": str(e), "trace": tb}

# Cache near-field compute to speed repeated identical queries from the UI
# compute_nearfield = lru_cache(maxsize=64)(compute_nearfield)
        

# ---------- plotting helpers ----------
# Build 3D visualization showing RIS plane + incident and reflected vectors
def build_vector_figure(data, texture_path="grid.png"):

    # Documentation describing purpose of this function
    """
    Build the 3D vector + RIS plane visualization.
    Attempts to load texture_path for plane texture; if missing, shows plain colored plane.
    """

    print("\n========== BUILD VECTOR FIGURE START ==========")

    # Extract incident vector from data dictionary
    # If not present, use default upward vector
    iv = data.get("incident_vector", np.array([0.0,0.0,1.0]))

    # Extract reflected vector from data dictionary
    # If not present, use default upward vector
    rv = data.get("reflected_vector", np.array([0.0,0.0,1.0]))

    # Print vectors for debugging
    print("Incident vector:", iv)
    print("Reflected vector:", rv)

    # Define half-width of RIS visualization plane
    plane_half = 1.3

    # Get RIS size (fixed 32x32)
    ny, nx = STATIC_RS2, STATIC_RS1

    # Ensure minimum grid resolution of 8x8 for visualization
    ny = max(ny, 8); nx = max(nx, 8)

    print("Plane grid resolution:", nx, "x", ny)

    # Generate evenly spaced Y-axis values for plane
    y_vals = np.linspace(-plane_half, plane_half, ny)

    # Generate evenly spaced Z-axis values for plane
    z_vals = np.linspace(-plane_half, plane_half, nx)

    # Create 2D grid for plane surface (Y-Z plane)
    Yp, Zp = np.meshgrid(y_vals, z_vals)

    # X coordinates are zero (plane lies on X=0)
    Xp = np.zeros_like(Yp)

    # Create empty Plotly figure
    fig = go.Figure()

    # Flag indicating whether texture successfully loaded
    tex_present = False

    # Check if texture image file exists
    if os.path.exists(texture_path):

        try:
            # Open texture image
            img = Image.open(texture_path).convert("RGBA")

            # Resize texture to match plane mesh resolution
            img_resized = img.resize((Yp.shape[1], Yp.shape[0]))

            # Convert image to NumPy array
            tex = np.asarray(img_resized)

            # Mark texture as available
            tex_present = True

            print("Texture loaded successfully:", texture_path)

        except Exception:

            # If any error occurs, disable texture
            tex_present = False

            print("Texture load failed. Using plain plane.")

    # If texture loaded successfully
    if tex_present:

        # Add surface representing RIS plane
        fig.add_trace(go.Surface(
            x=Xp, y=Yp, z=Zp,

            # Uniform surface color (texture not directly mapped in Plotly Surface)
            surfacecolor=np.ones_like(Xp),

            cmin=0, cmax=1,

            showscale=False,

            opacity=0.88,

            colorscale=[[0,'rgb(210,190,170)'], [1,'rgb(210,190,170)']],

            hoverinfo='skip',

            name='RIS plane'
        ))

    # If no texture
    else:

        # Add plain colored RIS plane
        fig.add_trace(go.Surface(
            x=Xp, y=Yp, z=Zp,
            surfacecolor=np.ones_like(Xp),
            showscale=False,
            opacity=0.78,
            colorscale=[[0,'rgb(210,190,170)'], [1,'rgb(210,190,170)']],
            hoverinfo='skip',
            name='RIS plane'
        ))

    # Draw grid lines on RIS plane (horizontal)
    for yv in y_vals:

        fig.add_trace(go.Scatter3d(
            x=[0,0],
            y=[yv,yv],
            z=[z_vals[0], z_vals[-1]],
            mode='lines',
            line=dict(color='saddlebrown', width=2),
            showlegend=False
        ))

    # Draw grid lines on RIS plane (vertical)
    for zv in z_vals:

        fig.add_trace(go.Scatter3d(
            x=[0,0],
            y=[y_vals[0], y_vals[-1]],
            z=[zv,zv],
            mode='lines',
            line=dict(color='saddlebrown', width=2),
            showlegend=False
        ))

    # Helper function to draw 3D arrow
    def add_arrow_line(fig, start, vec, color, name=None):

        # Convert inputs to numpy arrays
        start = np.array(start, dtype=float)
        vec = np.array(vec, dtype=float)

        # Compute vector magnitude
        norm = np.linalg.norm(vec)

        # If zero vector, assign default upward direction
        if norm < 1e-9:
            vec = np.array([0.,0.,1.])
            norm = 1.0

        # Normalize vector direction
        dir_u = vec / norm

        # Visible arrow length
        vis_len = 1.0

        # Compute arrow tip location
        tip = start + dir_u * vis_len

        # Compute shaft end location (slightly shorter)
        line_end = start + dir_u * (vis_len * 0.86)

        # Add arrow shaft
        fig.add_trace(go.Scatter3d(
            x=[start[0], line_end[0]],
            y=[start[1], line_end[1]],
            z=[start[2], line_end[2]],
            mode='lines',
            line=dict(color=color, width=6),
            name=name,
            hoverinfo='skip'
        ))

        # Add arrow head marker
        fig.add_trace(go.Scatter3d(
            x=[tip[0]],
            y=[tip[1]],
            z=[tip[2]],
            mode='markers',
            marker=dict(size=8, color=color, symbol='diamond'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Draw incident vector arrow
    add_arrow_line(fig, [1.5,0,0], -iv, 'red', name='Incident Vector')

    # Draw reflected vector arrow
    add_arrow_line(fig, [0,0,0], rv, 'blue', name='Reflected Vector')

    print("Vectors plotted.")

    # Configure 3D layout
    fig.update_layout(
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1.0, y=1.0, z=0.8),
            xaxis=dict(range=[-2,2], title='X',
                       backgroundcolor='rgb(245,245,245)', gridcolor='lightgray'),
            yaxis=dict(range=[-2,2], title='Y',
                       backgroundcolor='rgb(245,245,245)', gridcolor='lightgray'),
            zaxis=dict(range=[-2,2], title='Z',
                       backgroundcolor='rgb(245,245,245)', gridcolor='lightgray'),
            camera=dict(eye=dict(x=1.3, y=-1.4, z=0.8))
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True
    )

    print("========== BUILD VECTOR FIGURE END ==========\n")

    # Return completed figure
    return fig

# Build heatmap visualization of 1-bit RIS phase mask
def build_phase_figure(data):

    # Print start banner
    print("\n========== BUILD PHASE FIGURE START ==========")

    # Extract binary phase mask (shape RS2 x RS1)
    mask = np.array(data["mask"])

    # Print mask shape
    print("Mask shape:", mask.shape)

    # Convert binary mask (0 or 1) into degrees (0° or 180°)
    mask_deg = mask * 180.0

    # Flip vertically so visual orientation matches physical RIS layout
    mask_deg = np.flipud(mask_deg)

    # Extract number of columns (X direction elements)
    nx = mask.shape[1]

    # Extract number of rows (Y direction elements)
    ny = mask.shape[0]

    # Print mask statistics
    print("Unique mask states:", np.unique(mask))
    print("Number of 0° elements:", np.sum(mask == 0))
    print("Number of 180° elements:", np.sum(mask == 1))

    # Create X-axis element index labels (1 to nx)
    x = np.arange(1, nx+1)

    # Create Y-axis element index labels (1 to ny)
    y = np.arange(1, ny+1)

    # Create Plotly heatmap
    fig = go.Figure(
        go.Heatmap(

            # Phase values in degrees
            z=mask_deg,

            # X-axis values (Y element index)
            x=x,

            # Y-axis values (X element index)
            y=y,

            # Custom 2-color scale (Gold = 0°, Indigo = 180°)
            colorscale=[
                [0.0,"#FFD700"],
                [0.499,"#FFD700"],
                [0.5,"#4B0082"],
                [1.0,"#4B0082"]
            ],

            # Minimum phase value
            zmin=0,

            # Maximum phase value
            zmax=180,

            # Hide colorbar since only two discrete states
            showscale=False,

            # Custom hover information
            hovertemplate="Y=%{x}<br>X=%{y}<br>Phase=%{z}°<extra></extra>"
        )
    )

    # Configure layout settings
    fig.update_layout(

        # Title displayed above plot
        title="Randomized Phase Mask",

        # Label for X axis
        xaxis_title="Y-Elements",

        # Label for Y axis
        yaxis_title="X-Elements",

        # Adjust margins
        margin=dict(l=60,r=20,t=40,b=50)
    )

    # Print end banner
    print("========== BUILD PHASE FIGURE END ==========\n")

    # Return completed figure
    return fig

# Build heatmap visualization of per-element illumination magnitude
def build_illumination_figure(data):

    # Print start banner for debugging
    print("\n========== BUILD ILLUMINATION FIGURE START ==========")

    # Extract normalized illumination (in dB) from data dictionary
    im = np.array(data["im_norm"])

    # Print illumination array shape
    print("Illumination array shape:", im.shape)

    # Flip vertically so visualization matches physical RIS orientation
    im_vis = np.flipud(im)

    # Extract element X positions (in mm)
    x = data["x_vals_elements"]

    # Extract element Y positions (in mm)
    y = data["y_vals_elements"]

    # Print element coordinate info
    print("Element X range (mm):", (np.min(x), np.max(x)))
    print("Element Y range (mm):", (np.min(y), np.max(y)))

    # Print illumination statistics
    print("Illumination max (dB):", np.max(im))
    print("Illumination min (dB):", np.min(im))

    # Create Plotly heatmap figure
    fig = go.Figure(
        go.Heatmap(

            # Illumination values (normalized dB)
            z=im_vis,

            # X-axis values correspond to Y positions
            x=x,

            # Y-axis values correspond to X positions
            y=y,

            # Use continuous Viridis colormap
            colorscale="Viridis",

            # Add colorbar with title
            colorbar=dict(
                title="Magnitude dB",
                ticks="outside"
            ),

            # Custom hover display format
            hovertemplate="Y=%{x:.2f} mm<br>X=%{y:.2f} mm<br>dB=%{z:.2f}<extra></extra>"
        )
    )

    # Configure layout properties
    fig.update_layout(

        # Plot title
        title="Illumination Magnitude (per element, normalized dB)",

        # X-axis label
        xaxis_title="Y (mm)",

        # Y-axis label
        yaxis_title="X (mm)",

        # Adjust margins for spacing
        margin=dict(l=60,r=20,t=40,b=50)
    )

    # Print end banner
    print("========== BUILD ILLUMINATION FIGURE END ==========\n")

    # Return completed figure
    return fig

# Build X-Y plane heatmap (near-field cut at specified Z)
def build_xy_figure(data):

    # Print start banner
    print("\n========== BUILD XY FIGURE START ==========")

    # Extract clipped XY radiation pattern (dB)
    rr = np.array(data["rr_clip_xy"])

    # Extract X scanning positions (mm)
    x = data["x_pos"]

    # Extract Y scanning positions (mm)
    y = data["y_pos"]

    # Print grid dimensions
    print("XY grid shape:", rr.shape)
    print("X range (mm):", (np.min(x), np.max(x)))
    print("Y range (mm):", (np.min(y), np.max(y)))

    # Print radiation statistics
    print("XY max dB:", np.max(rr))
    print("XY min dB:", np.min(rr))

    # Create Plotly heatmap
    fig = go.Figure(
        go.Heatmap(

            # Transpose so axes match display orientation
            z=rr.T,

            # X-axis corresponds to Y positions
            x=y,

            # Y-axis corresponds to X positions
            y=x,

            # Use continuous Viridis colormap
            colorscale="Viridis",

            # Configure colorbar
            colorbar=dict(
                title="Magnitude (dB)",
                ticks="outside"
            ),

            # Custom hover display
            hovertemplate="Y=%{x} mm<br>X=%{y} mm<br>dB=%{z:.2f}<extra></extra>"
        )
    )

    # Retrieve dynamic range values from parameters dictionary
    dr1, dr2 = data.get("params", {}).get("dr", (-20,0))

    # Apply dynamic range limits to heatmap
    fig.data[0].zmin = float(dr1)
    fig.data[0].zmax = float(dr2)

    # Retrieve Z cut position
    zcut = data.get("params", {}).get("zcut", 0)

    # Retrieve receiver Z focus
    zr = data.get("params", {}).get("rx", (None,None,None))[2]

    # Configure layout
    fig.update_layout(

        # Title shows current cut plane and focus position
        title=f"X-Y Cut at Z = {zcut} mm at Z focus = {zr}",

        # Axis labels
        xaxis_title="Y (mm)",
        yaxis_title="X (mm)",

        # Margins
        margin=dict(l=60,r=40,t=50,b=50)
    )

    # Print end banner
    print("========== BUILD XY FIGURE END ==========\n")

    # Return figure
    return fig

# Build Y-Z plane heatmap (near-field cut at X = 0)
def build_yz_figure(data):

    # Print start banner
    print("\n========== BUILD YZ FIGURE START ==========")

    # Extract clipped YZ radiation pattern (dB)
    rr_yz = np.array(data["rr_yz_clip"])

    # Extract Z scanning positions (mm)
    zpos = data["z_pos"]

    # Extract Y scanning positions (mm)
    y = data["y_pos_yz"]

    # Print grid dimensions
    print("YZ grid shape:", rr_yz.shape)
    print("Z range (mm):", (np.min(zpos), np.max(zpos)))
    print("Y range (mm):", (np.min(y), np.max(y)))

    # Print radiation statistics
    print("YZ max dB:", np.max(rr_yz))
    print("YZ min dB:", np.min(rr_yz))

    # Create Plotly heatmap
    fig = go.Figure(
        go.Heatmap(

            # Transpose so axes match display orientation
            z=rr_yz.T,

            # X-axis corresponds to Y positions
            x=y,

            # Y-axis corresponds to Z positions
            y=zpos,

            # Use continuous Viridis colormap
            colorscale="Viridis",

            # Configure colorbar
            colorbar=dict(
                title="Magnitude (dB)",
                ticks="outside"
            ),

            # Custom hover display
            hovertemplate="Y=%{x} mm<br>Z=%{y} mm<br>dB=%{z:.2f}<extra></extra>"
        )
    )

    # Retrieve dynamic range values from parameters dictionary
    dr1, dr2 = data.get("params", {}).get("dr", (-20,0))

    # Apply dynamic range limits
    fig.data[0].zmin = float(dr1)
    fig.data[0].zmax = float(dr2)

    # Configure layout
    fig.update_layout(

        # Title for YZ cut
        title="Y-Z Cut at X = 0 mm",

        # Axis labels
        xaxis_title="Y (mm)",
        yaxis_title="Z (mm)",

        # Margins
        margin=dict(l=60,r=40,t=50,b=50)
    )

    # Print end banner
    print("========== BUILD YZ FIGURE END ==========\n")

    # Return figure
    return fig
