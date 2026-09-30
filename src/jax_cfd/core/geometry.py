import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple
from jax_cfd.core.mac_grid import MACGrid


def naca4_points(code: str, n_points: int, chord: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Lagrangian marker points for a symmetric NACA 4-digit airfoil.

    Uses cosine spacing for higher point density near leading/trailing edges.
    n_points is rounded up to the nearest even number.

    Args:
        code:     4-digit NACA code string, e.g. '0012'.
        n_points: Number of surface points (must be even; rounded up if odd).
        chord:    Chord length in physical units.

    Returns:
        (xs, ys) — numpy arrays of shape (n_points,) traversing the airfoil
        surface from the upper TE → LE → lower TE (counter-clockwise).
    """
    if n_points % 2 != 0:
        n_points += 1
    n_half = n_points // 2

    beta = np.linspace(0.0, np.pi, n_half + 1)
    x_c  = 0.5 * (1.0 - np.cos(beta))
    t    = int(str(code)[2:]) / 100.0
    yt   = 5.0 * t * (
        0.2969 * np.sqrt(x_c)
        - 0.1260 * x_c
        - 0.3516 * x_c ** 2
        + 0.2843 * x_c ** 3
        - 0.1015 * x_c ** 4
    )

    # Upper surface (LE→TE), then lower (TE→LE), dropping shared endpoints
    xs = np.concatenate([x_c[:-1], x_c[::-1][:-1]]) * chord
    ys = np.concatenate([yt[:-1],  -yt[::-1][:-1]]) * chord
    return xs, ys


def cylinder_coords_diff(
    radius,
    center,
    n_pts: int,
):
    """Differentiable cylinder geometry using pure JAX operations.

    Unlike ``get_geometry``, all inputs except *n_pts* are JAX-traced
    scalars so that ``jax.grad`` can propagate through the geometry.

    Parameters
    ----------
    radius : jnp scalar
        Cylinder radius (traced).
    center : tuple of two jnp scalars
        (cx, cy) centre coordinates (traced).
    n_pts : int
        Number of Lagrangian markers (static — determines array shape).

    Returns
    -------
    coords : (xs, ys) tuple of jnp arrays, each shape (n_pts,)
    ds_arr : jnp array shape (n_pts,)  — arc-length per marker.
    """
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_pts + 1)[:-1]
    xs = center[0] + radius * jnp.cos(theta)
    ys = center[1] + radius * jnp.sin(theta)
    ds_arr = jnp.full(n_pts, 2.0 * jnp.pi * radius / n_pts)
    return (xs, ys), ds_arr


def naca_coords_diff(
    thickness,
    camber,
    camber_pos,
    alpha,
    chord,
    center,
    n_pts: int,
):
    """Differentiable NACA 4-digit airfoil geometry using pure JAX operations.

    Parameters
    ----------
    thickness : jnp scalar  — max thickness as fraction of chord (e.g. 0.12).
    camber    : jnp scalar  — max camber as fraction of chord (0 for symmetric).
    camber_pos: jnp scalar  — chordwise position of max camber (e.g. 0.4).
    alpha     : jnp scalar  — angle of attack in radians.
    chord     : jnp scalar  — chord length.
    center    : tuple of two jnp scalars (cx, cy).
    n_pts     : int          — total Lagrangian markers (static, even).

    Returns
    -------
    coords : (xs, ys) tuple of jnp arrays, each shape (n_actual,)
    ds_arr : jnp array shape (n_actual,) — arc-length per marker.
    """
    n_half = n_pts // 2
    n_actual = 2 * n_half

    # 1. Cosine-spaced x/c in [0, 1]
    beta = jnp.linspace(0.0, jnp.pi, n_half + 1)
    x_c = 0.5 * (1.0 - jnp.cos(beta))  # (n_half+1,)

    # 2. Thickness distribution (NACA 4-digit)
    yt = 5.0 * thickness * (
        0.2969 * jnp.sqrt(jnp.maximum(x_c, 1e-14))
        - 0.1260 * x_c
        - 0.3516 * x_c ** 2
        + 0.2843 * x_c ** 3
        - 0.1015 * x_c ** 4
    )

    # 3. Camber line (piecewise via jnp.where)
    p_safe = jnp.maximum(camber_pos, 1e-6)
    yc_front = camber / p_safe ** 2 * (2.0 * p_safe * x_c - x_c ** 2)
    yc_rear = camber / jnp.maximum(1.0 - p_safe, 1e-6) ** 2 * (
        (1.0 - 2.0 * p_safe) + 2.0 * p_safe * x_c - x_c ** 2
    )
    yc = jnp.where(x_c <= p_safe, yc_front, yc_rear)

    # 4. Camber slope -> surface normal
    dyc_front = 2.0 * camber / p_safe ** 2 * (p_safe - x_c)
    dyc_rear = 2.0 * camber / jnp.maximum(1.0 - p_safe, 1e-6) ** 2 * (p_safe - x_c)
    dyc = jnp.where(x_c <= p_safe, dyc_front, dyc_rear)
    theta_c = jnp.arctan2(dyc, 1.0)

    # 5. Upper / lower surface coordinates
    xu = x_c - yt * jnp.sin(theta_c)
    yu = yc + yt * jnp.cos(theta_c)
    xl = x_c + yt * jnp.sin(theta_c)
    yl = yc - yt * jnp.cos(theta_c)

    # 6. Concatenate: upper TE->LE then lower LE->TE (CCW)
    xs_body = jnp.concatenate([xu[::-1][:-1], xl[1:]])  # (n_actual,)
    ys_body = jnp.concatenate([yu[::-1][:-1], yl[1:]])

    # 7. Center at mid-chord, scale by chord
    xs_body = (xs_body - 0.5) * chord
    ys_body = ys_body * chord

    # 8. Rotate by -alpha (positive alpha = nose up)
    cos_a = jnp.cos(-alpha)
    sin_a = jnp.sin(-alpha)
    xs_rot = xs_body * cos_a - ys_body * sin_a
    ys_rot = xs_body * sin_a + ys_body * cos_a

    # 9. Translate to center
    xs = xs_rot + center[0]
    ys = ys_rot + center[1]

    # 10. Arc-length per marker (wrap-around for TE closure)
    dxs = jnp.roll(xs, -1) - xs
    dys = jnp.roll(ys, -1) - ys
    ds_arr = jnp.sqrt(dxs ** 2 + dys ** 2 + 1e-14)

    return (xs, ys), ds_arr


def parsec_coords_diff(
    r_le, x_up, z_up, z_xx_up,
    x_lo, z_lo, z_xx_lo,
    z_te, dz_te, alpha_te, beta_te,
    alpha, chord, center, n_pts: int,
):
    """Differentiable PARSEC airfoil parameterization (11 shape params + AoA).

    Parameters
    ----------
    r_le      : jnp scalar — leading-edge radius.
    x_up      : jnp scalar — upper-crest chordwise position.
    z_up      : jnp scalar — upper-crest height.
    z_xx_up   : jnp scalar — upper-crest curvature (negative = convex up).
    x_lo      : jnp scalar — lower-crest chordwise position.
    z_lo      : jnp scalar — lower-crest depth (negative).
    z_xx_lo   : jnp scalar — lower-crest curvature (positive = concave up).
    z_te      : jnp scalar — trailing-edge vertical offset.
    dz_te     : jnp scalar — trailing-edge half-thickness.
    alpha_te  : jnp scalar — trailing-edge direction angle (rad).
    beta_te   : jnp scalar — trailing-edge wedge angle (rad).
    alpha     : jnp scalar — angle of attack (rad).
    chord     : jnp scalar — chord length.
    center    : tuple (cx, cy) of jnp scalars.
    n_pts     : int — total Lagrangian markers (static, even).

    Returns
    -------
    coords : (xs, ys) each shape (n_actual,)
    ds_arr : jnp array shape (n_actual,) — arc-length per marker.
    """
    n_half = n_pts // 2
    n_actual = 2 * n_half

    # 1. Cosine-spaced x/c in [0, 1]
    beta_ = jnp.linspace(0.0, jnp.pi, n_half + 1)
    x_c = 0.5 * (1.0 - jnp.cos(beta_))  # (n_half+1,)

    # 2. PARSEC basis: x^(n-1/2) for n=1..6
    #    powers = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    x_safe = jnp.maximum(x_c, 1e-14)
    sqrt_x = jnp.sqrt(x_safe)
    # Basis matrix: (n_half+1, 6)
    P = jnp.stack([
        sqrt_x,           # x^0.5
        x_safe * sqrt_x,  # x^1.5
        x_safe**2 * sqrt_x,  # x^2.5
        x_safe**3 * sqrt_x,  # x^3.5
        x_safe**4 * sqrt_x,  # x^4.5
        x_safe**5 * sqrt_x,  # x^5.5
    ], axis=1)

    # 3. Solve for upper-surface coefficients a1..a6
    r_le_safe = jnp.maximum(r_le, 1e-8)
    a1 = jnp.sqrt(2.0 * r_le_safe)
    x_up_safe = jnp.maximum(x_up, 0.05)

    # Precompute powers of x_up for conditions
    sqrt_xup = jnp.sqrt(x_up_safe)
    pows = jnp.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])

    # Build 5x5 system for a2..a6 (a1 fixed by r_LE)
    # Basis at x=1: [1, 1, 1, 1, 1, 1] → a1 + a2 + ... + a6 = z_te + dz_te
    # Condition 1: z(1) = z_te + dz_te
    # Condition 2: z'(1) = tan(alpha_te - beta_te/2)
    # Condition 3: z(x_up) = z_up
    # Condition 4: z'(x_up) = 0
    # Condition 5: z''(x_up) = z_xx_up

    # z'(x) = Σ (n-1/2) * a_n * x^(n-3/2)
    # z''(x) = Σ (n-1/2)*(n-3/2) * a_n * x^(n-5/2)
    # Row vectors at x=1 (x^anything = 1)
    row1_full = jnp.ones(6)            # z(1)
    row2_full = pows                    # z'(1) = Σ (n-1/2)*a_n

    xup_basis = jnp.array([
        sqrt_xup,
        x_up_safe * sqrt_xup,
        x_up_safe**2 * sqrt_xup,
        x_up_safe**3 * sqrt_xup,
        x_up_safe**4 * sqrt_xup,
        x_up_safe**5 * sqrt_xup,
    ])
    inv_sqrt_xup = 1.0 / jnp.maximum(sqrt_xup, 1e-8)
    xup_dbasis = jnp.array([
        0.5 * inv_sqrt_xup,                   # 0.5 * x^(-0.5)
        1.5 * sqrt_xup,                       # 1.5 * x^(0.5)
        2.5 * x_up_safe * sqrt_xup,           # 2.5 * x^(1.5)
        3.5 * x_up_safe**2 * sqrt_xup,        # 3.5 * x^(2.5)
        4.5 * x_up_safe**3 * sqrt_xup,        # 4.5 * x^(3.5)
        5.5 * x_up_safe**4 * sqrt_xup,        # 5.5 * x^(4.5)
    ])
    xup_ddbasis = jnp.array([
        -0.25 * inv_sqrt_xup / jnp.maximum(x_up_safe, 1e-8),  # -0.25 * x^(-1.5)
        0.75 * inv_sqrt_xup,                                    # 0.75 * x^(-0.5)
        3.75 * sqrt_xup,                                        # 3.75 * x^(0.5)
        8.75 * x_up_safe * sqrt_xup,                            # 8.75 * x^(1.5)
        15.75 * x_up_safe**2 * sqrt_xup,                        # 15.75 * x^(2.5)
        24.75 * x_up_safe**3 * sqrt_xup,                        # 24.75 * x^(3.5)
    ])

    # RHS for upper surface (subtract a1 contribution)
    z_te_up = z_te + dz_te
    zp_te_up = jnp.tan(alpha_te - beta_te / 2.0)

    rhs_up = jnp.array([
        z_te_up - a1,                     # z(1): Σa_n = z_te_up, minus a1
        zp_te_up - 0.5 * a1,             # z'(1): Σ(n-1/2)*a_n, minus 0.5*a1
        z_up - a1 * xup_basis[0],        # z(x_up), minus a1 term
        0.0 - a1 * xup_dbasis[0],        # z'(x_up)=0, minus a1 term
        z_xx_up - a1 * xup_ddbasis[0],   # z''(x_up), minus a1 term
    ])

    A_up = jnp.stack([
        row1_full[1:],          # z(1)
        row2_full[1:],          # z'(1)
        xup_basis[1:],          # z(x_up)
        xup_dbasis[1:],         # z'(x_up)
        xup_ddbasis[1:],        # z''(x_up)
    ])

    a_rest = jnp.linalg.solve(A_up, rhs_up)
    a_coeffs = jnp.concatenate([jnp.array([a1]), a_rest])

    # 4. Solve for lower-surface coefficients b1..b6
    b1 = -jnp.sqrt(2.0 * r_le_safe)
    x_lo_safe = jnp.maximum(x_lo, 0.05)

    sqrt_xlo = jnp.sqrt(x_lo_safe)
    inv_sqrt_xlo = 1.0 / jnp.maximum(sqrt_xlo, 1e-8)

    xlo_basis = jnp.array([
        sqrt_xlo,
        x_lo_safe * sqrt_xlo,
        x_lo_safe**2 * sqrt_xlo,
        x_lo_safe**3 * sqrt_xlo,
        x_lo_safe**4 * sqrt_xlo,
        x_lo_safe**5 * sqrt_xlo,
    ])
    xlo_dbasis = jnp.array([
        0.5 * inv_sqrt_xlo,
        1.5 * sqrt_xlo,
        2.5 * x_lo_safe * sqrt_xlo,
        3.5 * x_lo_safe**2 * sqrt_xlo,
        4.5 * x_lo_safe**3 * sqrt_xlo,
        5.5 * x_lo_safe**4 * sqrt_xlo,
    ])
    xlo_ddbasis = jnp.array([
        -0.25 * inv_sqrt_xlo / jnp.maximum(x_lo_safe, 1e-8),
        0.75 * inv_sqrt_xlo,
        3.75 * sqrt_xlo,
        8.75 * x_lo_safe * sqrt_xlo,
        15.75 * x_lo_safe**2 * sqrt_xlo,
        24.75 * x_lo_safe**3 * sqrt_xlo,
    ])

    z_te_lo = z_te - dz_te
    zp_te_lo = jnp.tan(alpha_te + beta_te / 2.0)

    rhs_lo = jnp.array([
        z_te_lo - b1,
        zp_te_lo - 0.5 * b1,
        z_lo - b1 * xlo_basis[0],
        0.0 - b1 * xlo_dbasis[0],
        z_xx_lo - b1 * xlo_ddbasis[0],
    ])

    A_lo = jnp.stack([
        row1_full[1:],
        row2_full[1:],
        xlo_basis[1:],
        xlo_dbasis[1:],
        xlo_ddbasis[1:],
    ])

    b_rest = jnp.linalg.solve(A_lo, rhs_lo)
    b_coeffs = jnp.concatenate([jnp.array([b1]), b_rest])

    # 5. Evaluate surfaces
    z_upper = P @ a_coeffs  # (n_half+1,)
    z_lower = P @ b_coeffs

    # 6. Concatenate: upper TE->LE then lower LE->TE (CCW)
    xs_body = jnp.concatenate([x_c[::-1][:-1], x_c[1:]])  # (n_actual,)
    zs_body = jnp.concatenate([z_upper[::-1][:-1], z_lower[1:]])

    # 7. Center at mid-chord, scale by chord
    xs_body = (xs_body - 0.5) * chord
    zs_body = zs_body * chord

    # 8. Rotate by -alpha (positive alpha = nose up)
    cos_a = jnp.cos(-alpha)
    sin_a = jnp.sin(-alpha)
    xs_rot = xs_body * cos_a - zs_body * sin_a
    ys_rot = xs_body * sin_a + zs_body * cos_a

    # 9. Translate to center
    xs = xs_rot + center[0]
    ys = ys_rot + center[1]

    # 10. Arc-length per marker (wrap-around)
    dxs = jnp.roll(xs, -1) - xs
    dys = jnp.roll(ys, -1) - ys
    ds_arr = jnp.sqrt(dxs ** 2 + dys ** 2 + 1e-14)

    return (xs, ys), ds_arr


def compute_normals_2d(coords_tuple):
    """Compute outward unit normals and tangents for a 2D closed curve.

    Assumes CCW-ordered points (e.g. upper TE -> LE -> lower TE for airfoils,
    or theta 0->2pi for cylinders).  The outward normal is the tangent rotated
    by -90 degrees: n = (ty, -tx).

    Parameters
    ----------
    coords_tuple : (xs, ys) tuple of jnp arrays, each (n_pts,)

    Returns
    -------
    normals  : (nx, ny) tuple of jnp arrays, each (n_pts,)
    tangents : (tx, ty) tuple of jnp arrays, each (n_pts,)
    """
    xs, ys = coords_tuple
    # Central-difference tangent with wrap-around
    dx = jnp.roll(xs, -1) - jnp.roll(xs, 1)
    dy = jnp.roll(ys, -1) - jnp.roll(ys, 1)
    mag = jnp.sqrt(dx ** 2 + dy ** 2 + 1e-14)
    tx, ty = dx / mag, dy / mag
    # Outward normal for CCW curve: rotate tangent by -90 deg
    nx, ny = ty, -tx
    return (nx, ny), (tx, ty)


def get_geometry(config: Dict[str, Any], grid: MACGrid) -> Tuple[Tuple[jnp.ndarray, ...], jnp.ndarray]:
    """
    Parses the 'immersed_body' block and returns Lagrangian marker positions (xs, ys)
    and their spacing (ds_arr).
    
    config examples:
    {'type': 'cylinder', 'radius': 0.5, 'center': [5.0, 5.0]}
    {'type': 'naca', 'code': '0012', 'chord': 1.0, 'center': [5.0, 5.0], 'alpha_deg': 15.0}
    {'type': 'stl', 'path': 'mesh.stl', 'scale': 1.0, 'center': [5.0, 5.0]}
    """
    gtype = config.get('type', 'none').lower()
    
    if gtype == 'none' or gtype is None:
        return tuple(jnp.empty(0) for _ in range(grid.ndim)), jnp.empty(0)
        
    dx = grid.spacings[0]
    dy = grid.spacings[1]
    
    if gtype == 'cylinder':
        if grid.ndim != 2:
            raise ValueError("Cylinder geometry is 2D-only. Use 'sphere' for 3D.")
        r = config.get('radius', 0.5)
        cx, cy = config.get('center', [0.0, 0.0])
        n_pts = int(2.0 * np.pi * r / min(dx, dy)) + 1
        theta = np.linspace(0, 2*np.pi, n_pts)[:-1]
        xs = cx + r * np.cos(theta)
        ys = cy + r * np.sin(theta)
        ds = 2.0 * np.pi * r / len(theta)
        ds_arr = np.full(len(theta), ds)
        return (jnp.array(xs), jnp.array(ys)), jnp.array(ds_arr)
        
    elif gtype == 'naca':
        code = config.get('code', '0012')
        chord = config.get('chord', 1.0)
        cx, cy = config.get('center', [0.0, 0.0])
        alpha = np.radians(config.get('alpha_deg', 0.0))
        n_pts = int(2.5 * chord / np.sqrt(dx*dy))
        if n_pts % 2 != 0: n_pts += 1
        
        # NACA 4-digit analytical generation
        n_half = n_pts // 2
        beta = np.linspace(0, np.pi, n_half + 1)
        x_c = 0.5 * (1.0 - np.cos(beta))
        t = int(code[2:]) / 100.0
        yt = 5.0 * t * (0.2969 * np.sqrt(x_c) - 0.1260 * x_c - 0.3516 * x_c**2 + 0.2843 * x_c**3 - 0.1015 * x_c**4)
        
        xu, yu = x_c, yt
        xl, yl = x_c, -yt
        
        xs_base = np.concatenate([xu[::-1], xl[1:]]) * chord
        ys_base = np.concatenate([yu[::-1], yl[1:]]) * chord
        
        # Center at 0.5 chord initially, then translate and rotate
        xs_base -= 0.5 * chord
        
        xs_rot = xs_base * np.cos(-alpha) - ys_base * np.sin(-alpha)
        ys_rot = xs_base * np.sin(-alpha) + ys_base * np.cos(-alpha)
        
        xs = xs_rot + cx
        ys = ys_rot + cy
        # Arc-length weights: exact distance between consecutive Lagrangian points
        dxs = np.diff(xs, append=xs[0])
        dys = np.diff(ys, append=ys[0])
        ds_arr = np.sqrt(dxs**2 + dys**2)
        return (jnp.array(xs), jnp.array(ys)), jnp.array(ds_arr)
        
    elif gtype == 'sphere':
        if grid.ndim != 3:
            raise ValueError("Sphere geometry is 3D-only. Use 'cylinder' for 2D.")
        r = config.get('radius', 0.5)
        center = config.get('center', [0.0, 0.0, 0.0])
        cx, cy, cz = center[0], center[1], center[2]

        dmin = min(grid.spacings)
        n_pts = max(int(4.0 * np.pi * r ** 2 / (dmin ** 2)), 200)

        # Fibonacci sphere sampling (near-uniform distribution)
        golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
        indices = np.arange(n_pts, dtype=np.float64)
        theta = 2.0 * np.pi * indices / golden_ratio
        phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / n_pts)

        xs = cx + r * np.sin(phi) * np.cos(theta)
        ys = cy + r * np.sin(phi) * np.sin(theta)
        zs = cz + r * np.cos(phi)

        ds = 4.0 * np.pi * r ** 2 / n_pts
        ds_arr = np.full(n_pts, ds)
        return (jnp.array(xs), jnp.array(ys), jnp.array(zs)), jnp.array(ds_arr)

    elif gtype == 'hemisphere':
        if grid.ndim != 3:
            raise ValueError("Hemisphere geometry is 3D-only.")
        r = config.get('radius', 0.5)
        center = config.get('center', [0.0, 0.0, 0.0])
        cx, cy, cz = center[0], center[1], center[2]
        wall_axis = config.get('wall_axis', 1)
        half = config.get('half', 'upper')

        dmin = min(grid.spacings)
        n_full = max(int(4.0 * np.pi * r ** 2 / (dmin ** 2)), 200)

        # Fibonacci sphere sampling (same as sphere)
        golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
        indices = np.arange(n_full, dtype=np.float64)
        theta = 2.0 * np.pi * indices / golden_ratio
        phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / n_full)

        # Local coords centered at origin
        local_x = r * np.sin(phi) * np.cos(theta)
        local_y = r * np.sin(phi) * np.sin(theta)
        local_z = r * np.cos(phi)
        local = np.column_stack([local_x, local_y, local_z])

        # Keep only upper or lower half along wall_axis
        if half == 'upper':
            mask = local[:, wall_axis] >= 0.0
        else:
            mask = local[:, wall_axis] <= 0.0
        local = local[mask]

        xs = local[:, 0] + cx
        ys = local[:, 1] + cy
        zs = local[:, 2] + cz

        n_pts = len(xs)
        ds = 2.0 * np.pi * r ** 2 / n_pts  # hemisphere area / n_pts
        ds_arr = np.full(n_pts, ds)
        return (jnp.array(xs), jnp.array(ys), jnp.array(zs)), jnp.array(ds_arr)

    elif gtype == 'box':
        if grid.ndim != 3:
            raise ValueError("Box geometry is 3D-only. Use 'rectangle' for 2D.")
        w = config.get('width', 1.0)    # x-extent
        h = config.get('height', 0.5)   # y-extent
        d = config.get('depth', 0.5)    # z-extent
        center = config.get('center', [0.0, 0.0, 0.0])
        cx, cy, cz = center[0], center[1], center[2]

        dmin = min(grid.spacings)
        nw = max(int(w / dmin), 2)
        nh = max(int(h / dmin), 2)
        nd = max(int(d / dmin), 2)

        all_x, all_y, all_z = [], [], []

        # Cell-centered sample points on each face (no edge duplicates)
        # ±y faces (floor/ceiling of box): nw × nd points
        xp = np.linspace(-w/2, w/2, nw, endpoint=False) + w / (2 * nw)
        zp = np.linspace(-d/2, d/2, nd, endpoint=False) + d / (2 * nd)
        xx_wd, zz_wd = np.meshgrid(xp, zp, indexing='ij')
        for sign in (-1, 1):
            all_x.append(xx_wd.ravel())
            all_y.append(np.full(nw * nd, sign * h / 2))
            all_z.append(zz_wd.ravel())

        # ±z faces (front/back): nw × nh points
        yp = np.linspace(-h/2, h/2, nh, endpoint=False) + h / (2 * nh)
        xx_wh, yy_wh = np.meshgrid(xp, yp, indexing='ij')
        for sign in (-1, 1):
            all_x.append(xx_wh.ravel())
            all_y.append(yy_wh.ravel())
            all_z.append(np.full(nw * nh, sign * d / 2))

        # ±x faces (left/right): nh × nd points
        yy_hd, zz_hd = np.meshgrid(yp, zp, indexing='ij')
        for sign in (-1, 1):
            all_x.append(np.full(nh * nd, sign * w / 2))
            all_y.append(yy_hd.ravel())
            all_z.append(zz_hd.ravel())

        xs = np.concatenate(all_x) + cx
        ys = np.concatenate(all_y) + cy
        zs = np.concatenate(all_z) + cz

        surface_area = 2.0 * (w * h + h * d + w * d)
        n_total = len(xs)
        ds_arr = np.full(n_total, surface_area / n_total)
        return (jnp.array(xs), jnp.array(ys), jnp.array(zs)), jnp.array(ds_arr)

    elif gtype == 'rectangle':
        if grid.ndim != 2:
            raise ValueError("Rectangle geometry is 2D-only.")
        w = config.get('width', 1.0)
        h = config.get('height', 0.5)
        cx, cy = config.get('center', [0.0, 0.0])
        angle = np.radians(config.get('angle_deg', 0.0))
        ds_target = min(dx, dy)
        n_w = max(int(w / ds_target), 2)
        n_h = max(int(h / ds_target), 2)
        # 4 edges CCW: bottom → right → top → left (endpoint=False to avoid duplicates)
        bottom = np.column_stack([np.linspace(-w/2,  w/2, n_w, endpoint=False), np.full(n_w, -h/2)])
        right  = np.column_stack([np.full(n_h,  w/2), np.linspace(-h/2,  h/2, n_h, endpoint=False)])
        top    = np.column_stack([np.linspace( w/2, -w/2, n_w, endpoint=False), np.full(n_w,  h/2)])
        left   = np.column_stack([np.full(n_h, -w/2), np.linspace( h/2, -h/2, n_h, endpoint=False)])
        pts = np.vstack([bottom, right, top, left])
        # Rotate around origin then translate
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot = pts[:, 0] * cos_a - pts[:, 1] * sin_a
        y_rot = pts[:, 0] * sin_a + pts[:, 1] * cos_a
        xs = x_rot + cx
        ys = y_rot + cy
        perimeter = 2.0 * (w + h)
        n_total = len(xs)
        ds_arr = np.full(n_total, perimeter / n_total)
        return (jnp.array(xs), jnp.array(ys)), jnp.array(ds_arr)

    elif gtype == 'wall_line':
        if grid.ndim != 2:
            raise ValueError("wall_line geometry is 2D-only.")
        start = np.array(config.get('start', [0.0, 0.0]), dtype=np.float64)
        end = np.array(config.get('end', [1.0, 0.0]), dtype=np.float64)
        length = float(np.linalg.norm(end - start))
        ds_target = min(dx, dy)
        n_pts = max(int(length / ds_target) + 1, 2)
        t_param = np.linspace(0, 1, n_pts)
        xs = start[0] + t_param * (end[0] - start[0])
        ys = start[1] + t_param * (end[1] - start[1])
        ds = length / max(n_pts - 1, 1)
        ds_arr = np.full(n_pts, ds)
        return (jnp.array(xs), jnp.array(ys)), jnp.array(ds_arr)

    elif gtype == 'wall_arc':
        if grid.ndim != 2:
            raise ValueError("wall_arc geometry is 2D-only.")
        cx, cy = config.get('center', [0.0, 0.0])
        r = config.get('radius', 1.0)
        theta_start = float(config.get('theta_start', 0.0))
        theta_end = float(config.get('theta_end', np.pi / 2))
        arc_length = abs(theta_end - theta_start) * r
        ds_target = min(dx, dy)
        n_pts = max(int(arc_length / ds_target) + 1, 2)
        theta = np.linspace(theta_start, theta_end, n_pts)
        xs = cx + r * np.cos(theta)
        ys = cy + r * np.sin(theta)
        ds = arc_length / max(n_pts - 1, 1)
        ds_arr = np.full(n_pts, ds)
        return (jnp.array(xs), jnp.array(ys)), jnp.array(ds_arr)

    elif gtype == 'polyline':
        if grid.ndim != 2:
            raise ValueError("polyline geometry is 2D-only.")
        verts = np.array(config.get('vertices', [[0.0, 0.0], [1.0, 0.0]]),
                         dtype=np.float64)
        closed = config.get('closed', False)
        if len(verts) < 2:
            raise ValueError("polyline requires at least 2 vertices.")
        # Build segments
        if closed:
            verts = np.vstack([verts, verts[0:1]])  # close the loop
        all_xs, all_ys, all_ds = [], [], []
        ds_target = min(dx, dy)
        for i in range(len(verts) - 1):
            p0, p1 = verts[i], verts[i + 1]
            seg_len = float(np.linalg.norm(p1 - p0))
            n_seg = max(int(seg_len / ds_target) + 1, 2)
            # Exclude last point to avoid duplicates at joints (except final segment)
            include_end = (i == len(verts) - 2)
            t_param = np.linspace(0, 1, n_seg + (0 if include_end else 1))
            if not include_end:
                t_param = t_param[:-1]
            xs_seg = p0[0] + t_param * (p1[0] - p0[0])
            ys_seg = p0[1] + t_param * (p1[1] - p0[1])
            ds_seg = seg_len / max(n_seg - 1, 1)
            all_xs.append(xs_seg)
            all_ys.append(ys_seg)
            all_ds.append(np.full(len(xs_seg), ds_seg))
        xs = np.concatenate(all_xs)
        ys = np.concatenate(all_ys)
        ds_arr = np.concatenate(all_ds)
        return (jnp.array(xs), jnp.array(ys)), jnp.array(ds_arr)

    elif gtype == 'stl':
        try:
            import trimesh
        except ImportError:
            raise ImportError("Please run `pip install trimesh` to use STL geometries.")
            
        path = config['path']
        scale = config.get('scale', 1.0)
        cx, cy = config.get('center', [0.0, 0.0])
        
        mesh = trimesh.load(path)
        # Assumes the STL is a 3D object and we extract a 2D slice at z=0 for 2D sims.
        # Future 3D update will directly ingest mesh vertices.
        if grid.ndim == 2:
            slice_2d = mesh.section(plane_origin=[0,0,0], plane_normal=[0,0,1])
            if slice_2d is None:
                raise ValueError(f"STL mesh at {path} does not intersect the z=0 plane.")
            
            # Extract ordered 2D boundary points
            path_2d, _ = slice_2d.to_planar()
            lines = path_2d.discrete
            
            # Combine discrete lines and resample them efficiently
            # For this prototype, we just flatten vertex lists
            pts = []
            for line in lines:
                pts.extend(line)
            pts = np.array(pts) * scale
            xs = pts[:, 0] + cx
            ys = pts[:, 1] + cy
            
            # Compute arc lengths roughly
            dxs = np.diff(xs, append=xs[0])
            dys = np.diff(ys, append=ys[0])
            ds_arr = np.sqrt(dxs**2 + dys**2)
            
            return (jnp.array(xs), jnp.array(ys)), jnp.array(ds_arr)
            
        elif grid.ndim == 3:
            # Full 3D Surface Sampling for genuine volumetric forcing
            area_scale = scale ** 2
            total_area = mesh.area * area_scale
            
            dx, dy, dz = grid.spacings
            cell_area = (dx*dy + dy*dz + dx*dz) / 3.0
            
            # Request enough points to satisfy Nyquist criteria (approx 2 points per cell face)
            n_pts = int(total_area / cell_area) * 2
            n_pts = max(n_pts, 100) # Safety
            
            pts, face_indices = trimesh.sample.sample_surface(mesh, n_pts)
            pts = pts * scale
            
            # The center parameter can be [x, y] or [x, y, z]
            c = config.get('center', [0.0, 0.0, 0.0])
            cx, cy, cz = c[0], c[1], c[2] if len(c) > 2 else 0.0
            
            xs = pts[:, 0] + cx
            ys = pts[:, 1] + cy
            zs = pts[:, 2] + cz
            
            ds_arr = np.full(n_pts, total_area / n_pts)
            
            return (jnp.array(xs), jnp.array(ys), jnp.array(zs)), jnp.array(ds_arr)
            
        else:
            raise NotImplementedError("3D STL volume surfacing is planned for the next expansion.")
            
    else:
        raise ValueError(f"Unsupported geometry type: {gtype}")


def make_roughness_array(
    roughness_type: str,
    radius: float,
    spacing: Tuple[float, float],
    n_elements: Tuple[int, int],
    wall_center: list,
    wall_axis: int = 1,
    half: str = 'upper',
) -> list:
    """Generate a regular grid array of roughness element body configs.

    Parameters
    ----------
    roughness_type : Geometry type for each element ('hemisphere', 'sphere', etc.).
    radius : Radius (height) of each roughness element.
    spacing : (ds0, ds1) spacing between element centers along the two
              in-plane axes (axes other than wall_axis).
    n_elements : (n0, n1) number of elements in each in-plane direction.
    wall_center : [x, y, z] center of the first element.
    wall_axis : Axis perpendicular to the wall (default 1 = y).
    half : 'upper' or 'lower' for hemisphere orientation.

    Returns
    -------
    List of body config dicts for use in domain.py ``'bodies'`` list.
    """
    in_plane = [ax for ax in range(3) if ax != wall_axis]
    ax0, ax1 = in_plane[0], in_plane[1]

    bodies = []
    for i in range(n_elements[0]):
        for j in range(n_elements[1]):
            center = list(wall_center)
            center[ax0] = wall_center[ax0] + i * spacing[0]
            center[ax1] = wall_center[ax1] + j * spacing[1]
            bodies.append({
                'type': roughness_type,
                'radius': radius,
                'center': center,
                'wall_axis': wall_axis,
                'half': half,
                'motion': {'type': 'static'},
            })
    return bodies


def make_bend_90(
    d: float = 1.0,
    R_c: float = 3.0,
    L_in: float = 5.0,
    L_out: float = 5.0,
) -> list:
    """Generate 5 IBM wall bodies for a 2D 90-degree bend channel.

    Flow enters horizontally (+x) along y in [0, d], turns 90 degrees
    through a curved section, and exits vertically (+y).

    Parameters
    ----------
    d     : Channel width (default 1.0).
    R_c   : Centerline curvature radius (default 3.0).
    L_in  : Straight inlet length before the bend (default 5.0).
    L_out : Straight outlet length after the bend (default 5.0).

    Returns
    -------
    List of 5 body config dicts for ``domain['bodies']``.
    """
    R_i = R_c - d / 2.0
    R_o = R_c + d / 2.0
    C = [L_in, d / 2.0 + R_c]
    x_out = C[0] + R_c

    return [
        # 1. Upper inlet wall
        {'type': 'wall_line', 'start': [0.0, d], 'end': [L_in, d],
         'motion': {'type': 'static'}},
        # 2. Inner arc (θ: -π/2 → 0)
        {'type': 'wall_arc', 'center': list(C), 'radius': R_i,
         'theta_start': -np.pi / 2, 'theta_end': 0.0,
         'motion': {'type': 'static'}},
        # 3. Outer arc (θ: -π/2 → 0)
        {'type': 'wall_arc', 'center': list(C), 'radius': R_o,
         'theta_start': -np.pi / 2, 'theta_end': 0.0,
         'motion': {'type': 'static'}},
        # 4. Left outlet wall
        {'type': 'wall_line', 'start': [x_out - d / 2, C[1]],
         'end': [x_out - d / 2, C[1] + L_out],
         'motion': {'type': 'static'}},
        # 5. Right outlet wall
        {'type': 'wall_line', 'start': [x_out + d / 2, C[1]],
         'end': [x_out + d / 2, C[1] + L_out],
         'motion': {'type': 'static'}},
    ]


def make_u_bend(
    d: float = 1.0,
    R_c: float = 2.0,
    L_in: float = 5.0,
    L_out: float = 5.0,
    gap: float = 0.0,
) -> list:
    """Generate IBM wall bodies for a 2D U-bend (180-degree turn).

    Flow enters horizontally (+x) at y in [gap+d, gap+2d], turns 180 degrees,
    and exits horizontally (-x) at y in [0, d].

    Parameters
    ----------
    d     : Channel width.
    R_c   : Centerline curvature radius.
    L_in  : Straight inlet length.
    L_out : Straight outlet length.
    gap   : Vertical gap between inlet and outlet legs (0 = adjacent).
    """
    _s = {'motion': {'type': 'static'}}
    R_i = R_c - d / 2.0
    R_o = R_c + d / 2.0
    y_bot = 0.0
    y_top = 2 * d + gap
    y_mid = d + gap / 2.0  # centerline of the bend
    cx = L_in  # arc center x

    bodies = [
        # Lower leg bottom wall
        {'type': 'wall_line', 'start': [0.0, y_bot], 'end': [L_out, y_bot], **_s},
        # Lower leg top wall
        {'type': 'wall_line', 'start': [0.0, d], 'end': [cx - R_i, d], **_s},
        # Upper leg bottom wall
        {'type': 'wall_line', 'start': [0.0, d + gap], 'end': [cx - R_i, d + gap], **_s},
        # Upper leg top wall
        {'type': 'wall_line', 'start': [0.0, y_top], 'end': [L_in, y_top], **_s},
        # Inner arc (180 degrees)
        {'type': 'wall_arc', 'center': [cx, y_mid], 'radius': R_i,
         'theta_start': -np.pi / 2, 'theta_end': np.pi / 2, **_s},
        # Outer arc (180 degrees)
        {'type': 'wall_arc', 'center': [cx, y_mid], 'radius': R_o,
         'theta_start': -np.pi / 2, 'theta_end': np.pi / 2, **_s},
    ]
    return bodies


def make_s_bend(
    d: float = 1.0,
    R_c: float = 2.0,
    L_in: float = 3.0,
    L_mid: float = 1.0,
    L_out: float = 3.0,
) -> list:
    """Generate IBM wall bodies for a 2D S-bend (double reverse curve).

    Flow enters horizontally at y=0..d, curves up by 2*R_c + L_mid, then
    exits horizontally.

    Parameters
    ----------
    d      : Channel width.
    R_c    : Centerline curvature radius for each 90-deg curve.
    L_in   : Straight inlet length.
    L_mid  : Straight length between the two bends.
    L_out  : Straight outlet length.
    """
    _s = {'motion': {'type': 'static'}}
    R_i = R_c - d / 2.0
    R_o = R_c + d / 2.0

    # First bend center
    cx1 = L_in
    cy1 = d / 2.0 + R_c
    # After first 90-deg bend, vertical segment
    x_vert = cx1 + R_c
    y_vert_end = cy1 + L_mid
    # Second bend center
    cx2 = x_vert
    cy2 = y_vert_end
    # After second 90-deg bend, horizontal exit at y = cy2 + R_c - d/2 .. cy2 + R_c + d/2
    x_exit_start = cx2 + R_c
    y_exit = cy2 + R_c

    bodies = [
        # Inlet bottom wall
        {'type': 'wall_line', 'start': [0.0, 0.0], 'end': [L_in, 0.0], **_s},
        # Inlet top wall
        {'type': 'wall_line', 'start': [0.0, d], 'end': [L_in, d], **_s},
        # First bend arcs (inner/outer, -pi/2 -> 0)
        {'type': 'wall_arc', 'center': [cx1, cy1], 'radius': R_i,
         'theta_start': -np.pi / 2, 'theta_end': 0.0, **_s},
        {'type': 'wall_arc', 'center': [cx1, cy1], 'radius': R_o,
         'theta_start': -np.pi / 2, 'theta_end': 0.0, **_s},
        # Vertical left wall
        {'type': 'wall_line', 'start': [x_vert - d / 2, cy1],
         'end': [x_vert - d / 2, cy2], **_s},
        # Vertical right wall
        {'type': 'wall_line', 'start': [x_vert + d / 2, cy1],
         'end': [x_vert + d / 2, cy2], **_s},
        # Second bend arcs (inner/outer, 0 -> pi/2)
        {'type': 'wall_arc', 'center': [cx2, cy2], 'radius': R_i,
         'theta_start': 0.0, 'theta_end': np.pi / 2, **_s},
        {'type': 'wall_arc', 'center': [cx2, cy2], 'radius': R_o,
         'theta_start': 0.0, 'theta_end': np.pi / 2, **_s},
        # Outlet bottom wall
        {'type': 'wall_line', 'start': [x_exit_start, y_exit - d / 2],
         'end': [x_exit_start + L_out, y_exit - d / 2], **_s},
        # Outlet top wall
        {'type': 'wall_line', 'start': [x_exit_start, y_exit + d / 2],
         'end': [x_exit_start + L_out, y_exit + d / 2], **_s},
    ]
    return bodies


def make_t_junction(
    d: float = 1.0,
    L_main: float = 10.0,
    L_branch: float = 5.0,
    branch_pos: float = 0.5,
) -> list:
    """Generate IBM wall bodies for a 2D T-junction.

    Horizontal main channel y in [0, d], vertical branch opens upward
    at fractional position *branch_pos* along the main channel.

    Parameters
    ----------
    d          : Channel width (same for main and branch).
    L_main     : Main channel total length.
    L_branch   : Branch length (upward from main channel top).
    branch_pos : Fractional position of branch center along main (0-1).
    """
    _s = {'motion': {'type': 'static'}}
    xb = L_main * branch_pos  # branch center x

    bodies = [
        # Main channel bottom wall (full length)
        {'type': 'wall_line', 'start': [0.0, 0.0], 'end': [L_main, 0.0], **_s},
        # Main channel top wall, left segment
        {'type': 'wall_line', 'start': [0.0, d], 'end': [xb - d / 2, d], **_s},
        # Main channel top wall, right segment
        {'type': 'wall_line', 'start': [xb + d / 2, d], 'end': [L_main, d], **_s},
        # Branch left wall
        {'type': 'wall_line', 'start': [xb - d / 2, d],
         'end': [xb - d / 2, d + L_branch], **_s},
        # Branch right wall
        {'type': 'wall_line', 'start': [xb + d / 2, d],
         'end': [xb + d / 2, d + L_branch], **_s},
    ]
    return bodies


def make_sudden_expansion(
    d_in: float = 1.0,
    d_out: float = 2.0,
    L_in: float = 5.0,
    L_out: float = 10.0,
) -> list:
    """Generate IBM wall bodies for a 2D sudden expansion (backward-facing step).

    Narrow channel (width d_in) expands to wide channel (width d_out).
    The expansion is symmetric about y = d_out/2.

    Parameters
    ----------
    d_in  : Inlet channel width.
    d_out : Outlet channel width (must be >= d_in).
    L_in  : Inlet length.
    L_out : Outlet length after expansion.
    """
    _s = {'motion': {'type': 'static'}}
    step_h = (d_out - d_in) / 2.0  # step height on each side

    bodies = [
        # Bottom wall: inlet at step_h, step down to 0
        {'type': 'polyline', 'vertices': [
            [0.0, step_h], [L_in, step_h], [L_in, 0.0],
            [L_in + L_out, 0.0]], 'closed': False, **_s},
        # Top wall: inlet at d_out - step_h, step up to d_out
        {'type': 'polyline', 'vertices': [
            [0.0, d_out - step_h], [L_in, d_out - step_h],
            [L_in, d_out], [L_in + L_out, d_out]], 'closed': False, **_s},
    ]
    return bodies


def make_channel_with_baffles(
    d: float = 1.0,
    L: float = 10.0,
    n_baffles: int = 3,
    baffle_len: float = 0.6,
    spacing: float = 0.0,
) -> list:
    """Generate IBM wall bodies for a 2D channel with alternating baffles.

    Baffles alternate between top and bottom walls. Useful for heat
    exchangers, CVD reactors, and mixing enhancement.

    Parameters
    ----------
    d          : Channel width.
    L          : Channel length.
    n_baffles  : Number of baffles.
    baffle_len : Baffle length as fraction of channel width d (0-1).
    spacing    : Gap between baffles (0 = auto-space evenly).
    """
    _s = {'motion': {'type': 'static'}}
    if spacing <= 0:
        spacing = L / (n_baffles + 1)

    bodies = []
    for i in range(n_baffles):
        x = spacing * (i + 1)
        bl = baffle_len * d
        if i % 2 == 0:
            # Bottom baffle (grows upward)
            bodies.append({
                'type': 'wall_line', 'start': [x, 0.0], 'end': [x, bl], **_s})
        else:
            # Top baffle (grows downward)
            bodies.append({
                'type': 'wall_line', 'start': [x, d], 'end': [x, d - bl], **_s})
    return bodies
