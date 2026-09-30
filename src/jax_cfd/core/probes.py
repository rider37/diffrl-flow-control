import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from typing import Dict, Tuple
from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.state import FluidState

def _coords_to_indices(coords: jnp.ndarray, axis_coords: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    """
    Converts physical coordinates (N_probes, ndim) into continuous fractional array indices (ndim, N_probes)
    for exact multilinear interpolation via map_coordinates.
    """
    indices = []
    for dim in range(coords.shape[1]):
        # Assuming uniform grid representation for each abstract 1D array
        dim_coords = axis_coords[dim]
        c_min = dim_coords[0]
        dc = dim_coords[1] - dim_coords[0]
        
        # Continuous mathematical index float
        idx = (coords[:, dim] - c_min) / dc
        indices.append(idx)
        
    return jnp.stack(indices, axis=0)

def extract_probe_data(state: FluidState, probe_coords: jnp.ndarray, grid: MACGrid) -> Dict[str, jnp.ndarray]:
    """
    Extracts fluid parameters (Pressure, Velocity) at arbitrary N-dimensional physical coordinates.
    Employs JAX's optimized `map_coordinates` to perform exact N-linear interpolation exactly matching
    the respective MAC grid stagger locations.
    
    Args:
        state: The current FluidState.
        probe_coords: Array of shape (N_probes, ndim) containing the physical query points.
        grid: The governing MAC grid.
        
    Returns:
        Dict: {'p': array(N_probes), 'u_0': array(N_probes), 'u_1': ...}
    """
    results = {}
    ndim = grid.ndim
    
    if probe_coords.shape[0] == 0:
        return results
        
    # Pressure (Cell Centers)
    p_indices = _coords_to_indices(probe_coords, grid.cell_centers())
    # order=1 ensures exact multi-linear interpolation
    # mode='nearest' provides edge replication for out-of-bound queries
    results['p'] = map_coordinates(state.pressure, p_indices, order=1, mode='nearest')
    
    # Velocities (Face Centers)
    for i, u_cmp in enumerate(state.velocity):
        u_indices = _coords_to_indices(probe_coords, grid.face_centers(i))
        results[f'u_{i}'] = map_coordinates(u_cmp, u_indices, order=1, mode='nearest')
        
    # Passive Scalars (Cell Centers) — skip non-field arrays (e.g. body_temps)
    for name, s_cmp in state.scalars.items():
        if s_cmp.ndim == grid.ndim:
            results[name] = map_coordinates(s_cmp, p_indices, order=1, mode='nearest')
        
    return results

def extract_ibm_forces(delta_U_lag: Tuple[jnp.ndarray, ...], ds_arr: jnp.ndarray, grid: MACGrid, dt: float) -> jnp.ndarray:
    """
    Extracts the globally integrated aerodynamic forces acting on the immersed body.
    
    Physical derivation (non-dimensionalized, rho=1):
      The IBM spreading computes a body force (acceleration) F_body[i,j] acting on the fluid.
      By Newton's 3rd Law, the reaction force on the body is:
        F_k^{body} = -sum_{Lagrangian k} (dU_k / dt) * ds_k
    where dU_k is the velocity deficiency required to satisfy no-slip, and ds_k is the
    representative arc-length/area weight for each Lagrangian marker.
    
    Args:
        delta_U_lag: Tuple of velocity corrections at Lagrangian points, per component.
        ds_arr:      Area/arc-length weight for each Lagrangian marker, shape (N_markers,).
        grid:        MAC grid (for dimensionality check).
        dt:          Sub-step time size (to convert velocity deficiency to force).
    
    Returns:
        forces: jnp.ndarray of shape (ndim,) - (Drag, Lift[, Thrust])
    """
    forces = []
    for dU in delta_U_lag:
        # Physical force = mass-flux correction, integrated over the body surface.
        # Sign: body force on fluid is +dU correction; reaction ON body is negative.
        F_total = -jnp.sum(dU * ds_arr) / dt
        forces.append(F_total)
        
    return jnp.array(forces)
