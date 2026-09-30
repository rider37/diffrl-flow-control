import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable
from jax_cfd.core.mac_grid import MACGrid

# Type for a boundary applicator function
BoundaryApplicator = Callable[[jnp.ndarray, int, bool, Dict[str, Any], MACGrid], jnp.ndarray]

# ── Wall-function constants ─────────────────────────────────────────────────
_KAPPA = 0.41       # von Kármán constant
_E     = 9.793      # log-law intercept (matches viscous sublayer at y+ ≈ 11.6)
_YPLUS_SWITCH = 11.6

def apply_boundaries(velocity: Tuple[jnp.ndarray, ...], boundaries_config: Dict[str, Any], grid: MACGrid, inlet_vel: Tuple = None) -> Tuple[jnp.ndarray, ...]:
    """
    Applies configured boundary conditions to the N-dimensional velocity field.

    Supports two config formats per face:

    Uniform (dict)::

        'x_min': {'type': 'inlet', 'velocity': [1.0, 0.0]}

    Segmented (list of dicts with ``range``)::

        'x_min': [
            {'type': 'wall_no_slip', 'range': [0.0, 0.3]},
            {'type': 'inlet', 'velocity': [1.0, 0.0], 'range': [0.3, 0.7]},
            {'type': 'wall_no_slip', 'range': [0.7, 1.0]},
        ]
    """
    axes_names = ['x', 'y', 'z']
    new_vel = list(velocity)

    for axis in range(grid.ndim):
        axis_name = axes_names[axis]

        for side, is_min in [('min', True), ('max', False)]:
            key = f"{axis_name}_{side}"
            if key not in boundaries_config:
                continue
            bc = boundaries_config[key]
            if isinstance(bc, list):
                # Segmented BC
                for cmp_idx in range(grid.ndim):
                    new_vel[cmp_idx] = _apply_segmented_bc(
                        new_vel[cmp_idx], cmp_idx, axis, is_min,
                        bc, grid, inlet_vel)
            else:
                # Uniform BC (original path)
                for cmp_idx in range(grid.ndim):
                    new_vel[cmp_idx] = _apply_bc_to_face(
                        new_vel[cmp_idx], cmp_idx, axis, is_min=is_min,
                        bc=bc, grid=grid, inlet_vel=inlet_vel)

    return tuple(new_vel)


# ── Segmented boundary helpers ────────────────────────────────────────────

def _along_face_coords(bnd_axis: int, cmp_idx: int, grid: MACGrid):
    """Return 1-D coordinate array along the face for mask building.

    For a face perpendicular to ``bnd_axis``, the "along-face" axes are all
    axes except ``bnd_axis``.  In 2-D there is exactly one such axis.

    The coordinate array must match the field shape along that axis.  On a MAC
    grid the shape depends on whether ``cmp_idx`` equals the along-face axis
    (face count = N+1) or not (cell count = N).
    """
    # In 2-D: bnd_axis=0 → along_axis=1; bnd_axis=1 → along_axis=0.
    along_axes = [a for a in range(grid.ndim) if a != bnd_axis]
    # For 2-D we only need the first (sole) along axis.
    along_axis = along_axes[0]
    if cmp_idx == along_axis:
        # Tangential component staggered along its own axis → N+1 faces
        return grid.face_coords_1d(along_axis)
    else:
        # Cell-centred along this axis → N centres
        return grid.cell_centers_1d(along_axis)


def _build_segment_mask(coords_1d, lo, hi):
    """Boolean mask: True where ``lo <= coord < hi`` (last segment uses <=)."""
    return (coords_1d >= lo) & (coords_1d <= hi)


def _apply_segmented_bc(field, cmp_idx, bnd_axis, is_min, segments, grid,
                         inlet_vel=None):
    """Apply segmented BCs on one velocity component / one face."""
    coords = _along_face_coords(bnd_axis, cmp_idx, grid)

    # Start from the field as-is; compose each segment via jnp.where
    result = field
    for seg in segments:
        lo, hi = seg['range']
        mask_1d = _build_segment_mask(coords, lo, hi)
        # Reshape mask for broadcasting: boundary face has shape along non-bnd axes
        shape = [1] * grid.ndim
        along_axis = [a for a in range(grid.ndim) if a != bnd_axis][0]
        shape[along_axis] = mask_1d.shape[0]
        mask = mask_1d.reshape(shape)

        # Compute what the uniform BC would produce on the entire face
        applied = _apply_bc_to_face(field, cmp_idx, bnd_axis, is_min,
                                    bc=seg, grid=grid, inlet_vel=inlet_vel)

        # Extract boundary slice index
        idx_bnd = [slice(None)] * grid.ndim
        idx_bnd[bnd_axis] = 0 if is_min else -1
        idx_bnd = tuple(idx_bnd)

        # Compose: where mask is True, use the segment's boundary value
        bnd_applied = applied[idx_bnd]
        bnd_current = result[idx_bnd]
        merged = jnp.where(mask.reshape(bnd_current.shape), bnd_applied, bnd_current)
        result = result.at[idx_bnd].set(merged)

    return result

def _apply_segmented_ghost_pad(padded_f, cmp_idx, bnd_axis, is_min, segments,
                                grid, inlet_vel=None):
    """Apply segmented ghost-cell padding for diffusion on one component/face.

    The padded array has +1 ghost cell on each side, so the along-face
    coordinate array must be taken from the *original* (unpadded) field, while
    the ghost/interior indices account for the padding offset.

    The ghost-cell slice along non-boundary axes has size N+2 (padded), but
    the segment mask is built from the original grid coords (size N or N+1).
    We pad the mask with False on both sides to match.
    """
    # Along-face coords in the *original* field (same size as interior)
    coords = _along_face_coords(bnd_axis, cmp_idx, grid)

    for seg in segments:
        lo, hi = seg['range']
        mask_1d = _build_segment_mask(coords, lo, hi)

        # Pad mask with False to match ghost-padded dimension (N → N+2)
        mask_1d = jnp.concatenate([jnp.array([False]), mask_1d, jnp.array([False])])

        # Compute what the uniform ghost pad would produce
        applied = _apply_ghost_pad(padded_f, cmp_idx, bnd_axis, is_min,
                                   bc=seg, grid=grid, inlet_vel=inlet_vel)

        # Ghost index in padded array
        idx_ghost = [slice(None)] * grid.ndim
        idx_ghost[bnd_axis] = 0 if is_min else -1
        idx_ghost = tuple(idx_ghost)

        ghost_applied = applied[idx_ghost]
        ghost_current = padded_f[idx_ghost]
        mask = mask_1d.reshape(ghost_current.shape)
        merged = jnp.where(mask, ghost_applied, ghost_current)
        padded_f = padded_f.at[idx_ghost].set(merged)

    return padded_f


def _apply_bc_to_face(field: jnp.ndarray, cmp_idx: int, bnd_axis: int, is_min: bool, bc: Dict[str, Any], grid: MACGrid, inlet_vel: Tuple = None) -> jnp.ndarray:
    """
    Applies a specific BC to `field` (the `cmp_idx` velocity component) on the `bnd_axis` face.
    If cmp_idx == bnd_axis, we are modifying the normal velocity component.
    If cmp_idx != bnd_axis, we are modifying the tangential velocity component.
    """
    bctype = bc.get('type', 'outlet_zero_gradient')
    
    # Generate indexing slice for the boundary face
    idx_bnd = [slice(None)] * grid.ndim
    idx_bnd[bnd_axis] = 0 if is_min else -1
    idx_bnd = tuple(idx_bnd)
    
    # Zero gradient (Neumann) -> copy from the adjacent inner layer
    if bctype == 'outlet_zero_gradient':
        idx_inner = list(idx_bnd)
        idx_inner[bnd_axis] = 1 if is_min else -2
        idx_inner = tuple(idx_inner)
        return field.at[idx_bnd].set(field[idx_inner])
        
    # Dirichlet (Fixed Value) -> set to exact value
    elif bctype == 'inlet' or bctype == 'fixedValue':
        val = inlet_vel[cmp_idx] if inlet_vel is not None else bc.get('velocity', [0.0]*grid.ndim)[cmp_idx]
        return field.at[idx_bnd].set(val)

    # Parabolic inlet (Poiseuille channel flow)
    elif bctype == 'inlet_parabolic':
        Um = bc.get('Um', 1.5)
        extent_axis = 1 - bnd_axis  # axis perpendicular to inlet face
        L = grid.domain_size[extent_axis]
        if cmp_idx == bnd_axis:  # normal (u): parabolic profile
            n = field.shape[extent_axis]
            if cmp_idx == extent_axis:
                # u-face grid aligned with extent axis → face positions
                dy = L / (n - 1) if n > 1 else L
                y = jnp.arange(n) * dy
            else:
                # u-face grid staggered → cell-center positions
                dy = L / n
                y = jnp.arange(n) * dy + dy / 2
            profile = Um * 4.0 * y * (L - y) / L**2
            return field.at[idx_bnd].set(profile)
        else:  # tangential (v): zero
            return field.at[idx_bnd].set(0.0)

    # No-slip Wall
    # - Normal component (cmp_idx == bnd_axis): the MAC face lies exactly on the
    #   wall, so setting it to 0 is the correct Dirichlet condition.
    # - Tangential component (cmp_idx != bnd_axis): the MAC face lies at a
    #   cell-centre position (half a cell spacing from the wall), NOT at the wall.
    #   Forcing this face value to 0 imposes no-slip at the wrong location and
    #   subsequently corrupts the diffusion ghost cells (ghost = -interior → 0).
    #   The correct approach is to enforce no-slip only through the antisymmetric
    #   ghost cell in pad_velocity_for_diffusion (ghost = -field[first_interior]),
    #   which places the zero exactly at the wall mid-point.
    elif bctype in ('wall_no_slip', 'wall_noslip'):
        if cmp_idx == bnd_axis:  # Normal: face is exactly at the wall → Dirichlet 0
            return field.at[idx_bnd].set(0.0)
        else:                    # Tangential: ghost-cell handles BC; leave face value
            return field
        
    # Slip Wall -> Normal velocity is 0, tangential is zero_gradient
    elif bctype in ('wall_slip',):
        if cmp_idx == bnd_axis: # Normal
            return field.at[idx_bnd].set(0.0)
        else: # Tangential (zero gradient)
            idx_inner = list(idx_bnd)
            idx_inner[bnd_axis] = 1 if is_min else -2
            idx_inner = tuple(idx_inner)
            return field.at[idx_bnd].set(field[idx_inner])

    # Wall function -> Normal is 0 (same as no-slip), tangential handled by ghost pad
    elif bctype == 'wall_function':
        if cmp_idx == bnd_axis:
            return field.at[idx_bnd].set(0.0)
        else:
            return field

    # Periodic -> Usually handled by grid padding, but we skip updates here for now
    elif bctype == 'periodic':
        return field

    else:
        raise ValueError(f"Unknown BC type '{bctype}' for component {cmp_idx}, axis {bnd_axis}. "
                         f"Valid types: outlet_zero_gradient, inlet, inlet_parabolic, fixedValue, "
                         f"wall_no_slip, wall_noslip, wall_slip, wall_function, periodic")

def _wall_function_ghost(u_interior, y1, nu, n_iter=10):
    """
    Compute ghost-cell values for the log-law wall function.

    Given the tangential velocity *u_interior* at wall distance *y1*,
    iteratively solve for the friction velocity u_τ using Newton's method
    on the log-law:  |u|/u_τ = (1/κ) ln(E y⁺),  y⁺ = y₁ u_τ / ν.

    For y⁺ < 11.6 the viscous-sublayer formula u_τ = √(ν|u|/y₁) is used
    instead (no iteration needed).

    The ghost value is set so that the wall-normal gradient of the padded
    field reproduces the modelled wall shear stress:

        ghost = u₁ − 2 y₁ u_τ² sign(u₁) / ν
    """
    u_mag = jnp.maximum(jnp.abs(u_interior), 1e-12)

    # ── Viscous-sublayer formula (always computed) ────────────────────
    u_tau_visc = jnp.sqrt(nu * u_mag / y1)

    # ── Newton iteration for log-law u_τ ─────────────────────────────
    u_tau = u_tau_visc
    for _ in range(n_iter):
        yplus = y1 * u_tau / nu
        f  = u_mag / u_tau - (1.0 / _KAPPA) * jnp.log(jnp.maximum(_E * yplus, 1.0))
        fp = -u_mag / (u_tau ** 2) - 1.0 / (_KAPPA * u_tau)
        u_tau = u_tau - f / fp
        u_tau = jnp.maximum(u_tau, 1e-12)

    # ── Switch: viscous sublayer if y⁺ < 11.6 ───────────────────────
    yplus_final = y1 * u_tau / nu
    u_tau = jnp.where(yplus_final < _YPLUS_SWITCH, u_tau_visc, u_tau)

    # ── Ghost cell ───────────────────────────────────────────────────
    tau_w = u_tau ** 2
    ghost = u_interior - 2.0 * y1 * tau_w * jnp.sign(u_interior) / nu
    return ghost


def pad_velocity_for_diffusion(velocity: Tuple[jnp.ndarray, ...], boundaries_config: Dict[str, Any], grid: MACGrid, inlet_vel: Tuple = None) -> Tuple[jnp.ndarray, ...]:
    """
    Creates exact Ghost-Cell paddings for the Laplacian diff_axis_padded operator.
    Returns appropriately padded velocity tensors (size + 2 along each axis), where
    the ghost values satisfy Dirichlet/Neumann derivatives according to the BC configs.
    """
    axes_names = ['x', 'y', 'z']
    padded_vel = []
    
    for cmp_idx, field in enumerate(velocity):
        # We need to pad `field` by 1 on both sides for each axis
        pad_widths = [(1, 1) for _ in range(grid.ndim)]
        
        # First perform a generic zero pad, then we will overwrite the ghost boundaries
        padded_f = jnp.pad(field, pad_widths, mode='constant')
        
        for bnd_axis in range(grid.ndim):
            axis_name = axes_names[bnd_axis]

            for side, is_min in [('min', True), ('max', False)]:
                key = f"{axis_name}_{side}"
                bc = boundaries_config.get(key, {'type': 'outlet_zero_gradient'})
                if isinstance(bc, list):
                    padded_f = _apply_segmented_ghost_pad(
                        padded_f, cmp_idx, bnd_axis, is_min, bc, grid, inlet_vel)
                else:
                    padded_f = _apply_ghost_pad(
                        padded_f, cmp_idx, bnd_axis, is_min=is_min,
                        bc=bc, grid=grid, inlet_vel=inlet_vel)
            
        padded_vel.append(padded_f)
        
    return tuple(padded_vel)

def _apply_ghost_pad(padded_f: jnp.ndarray, cmp_idx: int, bnd_axis: int,
                     is_min: bool, bc: Dict[str, Any],
                     grid: MACGrid = None, inlet_vel: Tuple = None) -> jnp.ndarray:
    """Sets the ghost values (index 0 or -1 in the padded array) based on MAC logic."""
    bctype = bc.get('type', 'outlet_zero_gradient')

    ndim = padded_f.ndim
    idx_ghost = [slice(None)] * ndim
    idx_ghost[bnd_axis] = 0 if is_min else -1
    idx_ghost = tuple(idx_ghost)

    idx_first_interior = [slice(None)] * ndim
    idx_first_interior[bnd_axis] = 1 if is_min else -2
    idx_first_interior = tuple(idx_first_interior)

    # 1. Neumann boundaries (Outlet/Slip Tangent) -> Copy interior value (Zero Gradient)
    if bctype == 'outlet_zero_gradient':
        # ghost = interior
        return padded_f.at[idx_ghost].set(padded_f[idx_first_interior])

    # 2. No-Slip Wall
    elif bctype in ('wall_no_slip', 'wall_noslip'):
        if cmp_idx == bnd_axis: # Normal
            return padded_f.at[idx_ghost].set(padded_f[idx_first_interior])
        else: # Tangential
            return padded_f.at[idx_ghost].set(-padded_f[idx_first_interior])

    # 3. Slip Wall
    elif bctype == 'wall_slip':
        if cmp_idx == bnd_axis: # Normal
            return padded_f.at[idx_ghost].set(padded_f[idx_first_interior])
        else: # Tangential
            return padded_f.at[idx_ghost].set(padded_f[idx_first_interior])

    # 4. Dirichlet (Inlet / Fixed Value)
    elif bctype == 'inlet' or bctype == 'fixedValue':
        val = inlet_vel[cmp_idx] if inlet_vel is not None else bc.get('velocity', [0.0]*ndim)[cmp_idx]
        if cmp_idx == bnd_axis: # Normal
            return padded_f.at[idx_ghost].set(padded_f[idx_first_interior])
        else: # Tangential
            return padded_f.at[idx_ghost].set(2.0 * val - padded_f[idx_first_interior])

    # 4b. Parabolic inlet (Poiseuille)
    elif bctype == 'inlet_parabolic':
        if cmp_idx == bnd_axis:  # Normal: Neumann ghost (∂u/∂n = 0)
            return padded_f.at[idx_ghost].set(padded_f[idx_first_interior])
        else:  # Tangential: antisymmetric (v=0 at inlet face)
            return padded_f.at[idx_ghost].set(-padded_f[idx_first_interior])

    # 5. Periodic
    elif bctype == 'periodic':
        src = [slice(None)] * ndim
        src[bnd_axis] = -2 if is_min else 1
        return padded_f.at[tuple(idx_ghost)].set(padded_f[tuple(src)])

    # 6. Wall function (log-law wall model)
    elif bctype == 'wall_function':
        if cmp_idx == bnd_axis:  # Normal → same as no-slip
            return padded_f.at[idx_ghost].set(padded_f[idx_first_interior])
        else:  # Tangential → wall-function ghost cell
            nu = bc['nu']  # Python float, set by runner
            # Wall distance: distance from wall to first tangential face centre.
            # For MAC grid, tangential velocity is cell-centred along bnd_axis,
            # so y₁ = first (or last) cell-centre coordinate.
            # NOTE: avoid float() — this runs inside jax.lax.scan (traced).
            cc = grid.cell_centers_1d(bnd_axis)
            y1 = cc[0] if is_min else (grid.domain_size[bnd_axis] - cc[-1])
            ghost = _wall_function_ghost(padded_f[idx_first_interior], y1, nu)
            return padded_f.at[idx_ghost].set(ghost)

    # Fallback to copy
    return padded_f.at[idx_ghost].set(padded_f[idx_first_interior])


def pad_scalar_for_diffusion(
    scalar: jnp.ndarray,
    scalar_name: str,
    boundaries_config: Dict[str, Any],
    grid
) -> jnp.ndarray:
    """
    Creates Ghost-Cell padding for scalar diffusion (Laplacian) with proper BC.

    For each boundary face, reads the scalar boundary condition from config.
    Config BC entry supports an optional scalar-value key, e.g.:
        'y_min': {'type': 'wall_noslip', 'T': 1.0}   # Dirichlet T=1 at bottom wall
        'y_max': {'type': 'wall_noslip', 'T': 0.0}   # Dirichlet T=0 at top wall
        'x_min': {'type': 'inlet'}                   # Neumann (∂T/∂n=0) as fallback

    Ghost-Cell Dirichlet: ghost = 2*T_wall - interior_1
      → (ghost + interior_1) / 2 = T_wall   ✅ exact face value = T_wall.

    Ghost-Cell Neumann (∂T/∂n=0): ghost = interior_1
      → (interior_1 - ghost) / dx = 0       ✅ zero gradient.

    Returns:
        padded: array of shape (N+2, ...) padded by 1 ghost layer on each side.
    """
    axes_names = ['x', 'y', 'z']
    ndim = grid.ndim

    # Initial zero-pad (ghost cells will be overwritten below)
    pad_widths = [(1, 1) for _ in range(ndim)]
    padded = jnp.pad(scalar, pad_widths, mode='edge')  # Neumann default

    for bnd_axis in range(ndim):
        axis_name = axes_names[bnd_axis]

        for is_min in (True, False):
            side = 'min' if is_min else 'max'
            bc_key = f"{axis_name}_{side}"
            bc = boundaries_config.get(bc_key, {})

            # Ghost and interior indices in the padded array
            idx_ghost = [slice(None)] * ndim
            idx_ghost[bnd_axis] = 0 if is_min else -1

            idx_interior = [slice(None)] * ndim
            idx_interior[bnd_axis] = 1 if is_min else -2

            idx_ghost = tuple(idx_ghost)
            idx_interior = tuple(idx_interior)

            if isinstance(bc, list):
                # Segmented scalar BC
                padded = _apply_segmented_scalar_ghost(
                    padded, scalar_name, bnd_axis, is_min, bc, grid)
            else:
                bctype = bc.get('type', '')
                if bctype == 'periodic':
                    src = [slice(None)] * ndim
                    src[bnd_axis] = -2 if is_min else 1
                    padded = padded.at[idx_ghost].set(padded[tuple(src)])
                elif scalar_name in bc:
                    T_wall = float(bc[scalar_name])
                    ghost_val = 2.0 * T_wall - padded[idx_interior]
                    padded = padded.at[idx_ghost].set(ghost_val)
                else:
                    pass  # Neumann: already correct from edge pad

    return padded


def _apply_segmented_scalar_ghost(padded, scalar_name, bnd_axis, is_min,
                                   segments, grid):
    """Segmented ghost-cell padding for scalar diffusion."""
    ndim = grid.ndim
    # Along-face coords (scalar is cell-centred → always cell_centers_1d)
    along_axes = [a for a in range(ndim) if a != bnd_axis]
    along_axis = along_axes[0]
    coords = grid.cell_centers_1d(along_axis)

    idx_ghost = [slice(None)] * ndim
    idx_ghost[bnd_axis] = 0 if is_min else -1
    idx_ghost = tuple(idx_ghost)

    idx_interior = [slice(None)] * ndim
    idx_interior[bnd_axis] = 1 if is_min else -2
    idx_interior = tuple(idx_interior)

    for seg in segments:
        lo, hi = seg['range']
        mask_1d = _build_segment_mask(coords, lo, hi)
        # Pad mask with False to match ghost-padded dimension (N → N+2)
        mask_1d = jnp.concatenate([jnp.array([False]), mask_1d, jnp.array([False])])

        bctype = seg.get('type', '')
        if bctype == 'periodic':
            src = [slice(None)] * ndim
            src[bnd_axis] = -2 if is_min else 1
            ghost_seg = padded[tuple(src)]
        elif scalar_name in seg:
            T_wall = float(seg[scalar_name])
            ghost_seg = 2.0 * T_wall - padded[idx_interior]
        else:
            ghost_seg = padded[idx_interior]  # Neumann

        ghost_current = padded[idx_ghost]
        mask = mask_1d.reshape(ghost_current.shape)
        merged = jnp.where(mask, ghost_seg, ghost_current)
        padded = padded.at[idx_ghost].set(merged)

    return padded
