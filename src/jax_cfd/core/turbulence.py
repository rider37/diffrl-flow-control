import jax
import jax.numpy as jnp
from typing import Callable, Dict, Any, Tuple
from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.state import FluidState
from jax_cfd.core.operators import velocity_gradient_tensor

# Type alias for a turbulence viscosity closure function
ViscosityClosureFn = Callable[[FluidState, MACGrid, Dict[str, Any]], jnp.ndarray]

def get_turbulence_model(model_name: str, **params) -> ViscosityClosureFn:
    """
    Factory builder for turbulence closures.

    For models that accept coefficients (smagorinsky, wale) pass them as
    keyword arguments, e.g. get_turbulence_model('smagorinsky', Cs=0.32).
    """
    name = model_name.lower()
    if name == 'laminar':
        return get_laminar_viscosity
    elif name == 'wale':
        Cw = params.get('Cw', 0.32)
        return lambda state, grid, p: get_wale_viscosity(state, grid, {**p, 'Cw': Cw})
    elif name == 'smagorinsky':
        return make_smagorinsky(Cs=params.get('Cs', 0.1))
    elif name == 'dynamic_smagorinsky':
        return make_dynamic_smagorinsky(
            Cs_max=params.get('Cs_max', 0.3),
            periodic_axes=tuple(params.get('periodic_axes', ())),
        )
    elif name == 'neural':
        closure_fn, _ = make_neural_closure(
            hidden_dim=params.get('hidden_dim', 32),
            n_layers=params.get('n_layers', 2),
            params=params.get('weights', None),
        )
        return closure_fn
    else:
        raise ValueError(
            f"Unknown turbulence model '{model_name}'. "
            f"Choose from: laminar, wale, smagorinsky, dynamic_smagorinsky, neural"
        )


def make_smagorinsky(Cs: float = 0.1) -> ViscosityClosureFn:
    """
    Returns a Smagorinsky SGS closure with the given model constant Cs.

    The eddy viscosity is:
        ν_t = (Cs · Δ)² · |S|
    where Δ = (∏ dx_i)^(1/ndim) is the filter width and
    |S| = sqrt(2 S_ij S_ij) is the strain-rate magnitude.
    """
    def _smagorinsky(state: FluidState, grid: MACGrid, params: Dict[str, Any]) -> jnp.ndarray:
        ndim = grid.ndim
        delta = grid.cell_volumes() ** (1.0 / ndim)

        g = velocity_gradient_tensor(state.velocity, grid)
        g_t = jnp.transpose(g, axes=(1, 0) + tuple(range(2, 2 + ndim)))
        S = 0.5 * (g + g_t)
        S_sq = jnp.einsum('ij...,ij...->...', S, S)

        # nu_t = (Cs·Δ)² · sqrt(2 · S_ij S_ij)
        return (Cs * delta) ** 2 * jnp.sqrt(jnp.maximum(2.0 * S_sq, 0.0))

    return _smagorinsky

def get_laminar_viscosity(state: FluidState, grid: MACGrid, params: Dict[str, Any]) -> jnp.ndarray:
    """Returns zero turbulent viscosity (laminar flow)."""
    return jnp.zeros(grid.cell_shape())

def get_wale_viscosity(state: FluidState, grid: MACGrid, params: Dict[str, Any]) -> jnp.ndarray:
    """
    Computes Wall-Adapting Local Eddy-viscosity (WALE) for N-dimensions.
    Extracts the velocity gradient tensor g_{ij} = du_i / dx_j at cell centers,
    and constructs the invariants S_{ij}, O_{ij}, and Sd_{ij} using Einstein summation.
    """
    ndim = grid.ndim
    Cw = params.get('Cw', 0.32)

    # Velocity gradient tensor g[i,j,...] = du_i/dx_j  (shared with Smagorinsky)
    g = velocity_gradient_tensor(state.velocity, grid)
    
    # Strain rate S_{ij} = 0.5 * (g_{ij} + g_{ji})
    g_t = jnp.transpose(g, axes=(1, 0) + tuple(range(2, 2 + ndim)))
    S = 0.5 * (g + g_t)
    S_sq = jnp.einsum('ij...,ij...->...', S, S)

    # Traceless symmetric part of g²: Sd_{ij} = 0.5*(g²+g²ᵀ) - (1/ndim)*δ_{ij}*tr(g²)
    # g2_{ij} = g_{ik} g_{kj}
    g2 = jnp.einsum('ik...,kj...->ij...', g, g)
    g2_t = jnp.transpose(g2, axes=(1, 0) + tuple(range(2, 2 + ndim)))
    
    # Trace of g2 is g2_{kk}
    g2_trace = jnp.einsum('ii...->...', g2)
    
    # Kronecker delta expanded to grid
    delta = jnp.eye(ndim)
    delta_shape = delta.shape + (1,) * ndim
    delta_grid = jnp.reshape(delta, delta_shape)
    
    Sd = 0.5 * (g2 + g2_t) - (1.0 / ndim) * delta_grid * g2_trace
    
    Sd_sq = jnp.einsum('ij...,ij...->...', Sd, Sd)
    
    # Filter width scale (cell-local for non-uniform grids)
    filter_width = grid.cell_volumes() ** (1.0 / ndim)
    
    # WALE Numerator/Denominator (Nicoud & Ducros 1999)
    num = (Sd_sq) ** (3.0 / 2.0)
    den = (S_sq) ** (5.0 / 2.0) + (Sd_sq) ** (5.0 / 4.0)
    
    # Prevent div by zero
    num = jnp.where(den < 1e-10, 0.0, num / den)
    
    nu_t = (Cw * filter_width) ** 2 * num
    return nu_t


# ── Dynamic Smagorinsky helpers ──────────────────────────────────────────

def _box_filter_nd(
    field: jnp.ndarray,
    ndim: int,
    periodic_axes: Tuple[int, ...] = (),
) -> jnp.ndarray:
    """Top-hat (box) filter at 2Δ scale — 3-point average along each axis.

    This is the standard test filter for the Germano identity with
    filter-width ratio α = 2.
    """
    for axis in range(ndim):
        pw = [(0, 0)] * ndim
        pw[axis] = (1, 1)
        mode = 'wrap' if axis in periodic_axes else 'edge'
        padded = jnp.pad(field, pw, mode=mode)
        s0 = [slice(None)] * ndim
        s1 = [slice(None)] * ndim
        s2 = [slice(None)] * ndim
        s0[axis] = slice(None, -2)
        s1[axis] = slice(1, -1)
        s2[axis] = slice(2, None)
        field = (padded[tuple(s0)] + padded[tuple(s1)] + padded[tuple(s2)]) / 3.0
    return field


def _face_to_cell_center(u_face: jnp.ndarray, axis: int, ndim: int) -> jnp.ndarray:
    """Average a face-centred velocity component to cell centres along its stagger axis."""
    sl_p = [slice(None)] * ndim
    sl_m = [slice(None)] * ndim
    sl_p[axis] = slice(1, None)
    sl_m[axis] = slice(None, -1)
    return 0.5 * (u_face[tuple(sl_p)] + u_face[tuple(sl_m)])


def _strain_from_vel_cc(
    u_cc: Tuple[jnp.ndarray, ...],
    grid: MACGrid,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute strain-rate tensor S_ij and its magnitude |S| from cell-centred velocity.

    Returns (S, S_mag) where S has shape (ndim, ndim, *cell_shape) and
    S_mag has shape cell_shape.
    """
    from jax_cfd.core.operators import _central_denom_broadcast
    ndim = grid.ndim
    # Gradient tensor at cell centres via central differences
    g_list = []
    for i in range(ndim):
        grad_ui = []
        for j in range(ndim):
            pw = [(0, 0)] * ndim
            pw[j] = (1, 1)
            u_pad = jnp.pad(u_cc[i], pw, mode='edge')
            sl_p = [slice(None)] * ndim
            sl_m = [slice(None)] * ndim
            sl_p[j] = slice(2, None)
            sl_m[j] = slice(None, -2)
            denom = _central_denom_broadcast(grid, j)
            grad_ui.append((u_pad[tuple(sl_p)] - u_pad[tuple(sl_m)]) / denom)
        g_list.append(jnp.stack(grad_ui, axis=0))
    g = jnp.stack(g_list, axis=0)  # (ndim, ndim, *cell_shape)
    g_t = jnp.transpose(g, axes=(1, 0) + tuple(range(2, 2 + ndim)))
    S = 0.5 * (g + g_t)
    S_mag = jnp.sqrt(jnp.maximum(2.0 * jnp.einsum('ij...,ij...->...', S, S), 0.0))
    return S, S_mag


def make_dynamic_smagorinsky(
    Cs_max: float = 0.3,
    periodic_axes: Tuple[int, ...] = (),
) -> ViscosityClosureFn:
    """
    Dynamic Smagorinsky SGS model (Germano 1991, Lilly 1992).

    The model coefficient Cs is computed locally at each cell and time step
    using the Germano identity with a test filter at 2Δ.  Lilly's
    least-squares contraction avoids tensor inversion::

        L_ij = hat(u_i u_j) - hat(u_i) hat(u_j)
        M_ij = 2Δ²( hat(|S| S_ij) - α² |hat(S)| hat(S)_ij )
        Cs²  = clamp( L_ij M_ij / (M_ij M_ij + ε),  0,  Cs_max² )
        ν_t  = Cs² Δ² |S|

    where hat(·) is the test filter (box filter at 2Δ, α = 2).

    Negative Cs² (backscatter) is clipped to zero for stability.
    """
    alpha_sq = 4.0  # (test_filter_width / grid_filter_width)² = (2Δ/Δ)² = 4

    def _dynamic_smagorinsky(
        state: FluidState,
        grid: MACGrid,
        params: Dict[str, Any],
    ) -> jnp.ndarray:
        ndim = grid.ndim
        vel = state.velocity
        delta = grid.cell_volumes() ** (1.0 / ndim)  # grid filter width

        # ── Grid-scale strain from original velocity ──────────────────
        g = velocity_gradient_tensor(vel, grid)
        g_t = jnp.transpose(g, axes=(1, 0) + tuple(range(2, 2 + ndim)))
        S = 0.5 * (g + g_t)
        S_mag = jnp.sqrt(jnp.maximum(2.0 * jnp.einsum('ij...,ij...->...', S, S), 0.0))

        # ── Cell-centred velocities ───────────────────────────────────
        u_cc = tuple(_face_to_cell_center(vel[i], i, ndim) for i in range(ndim))

        # ── Test-filtered cell-centred velocities ─────────────────────
        u_hat = tuple(_box_filter_nd(u_cc[i], ndim, periodic_axes) for i in range(ndim))

        # ── Test-filtered strain |hat(S)|, hat(S)_ij ─────────────────
        S_hat, S_hat_mag = _strain_from_vel_cc(u_hat, grid)

        # ── L_ij = hat(u_i u_j) - hat(u_i) hat(u_j) ─────────────────
        # M_ij = 2Δ²( hat(|S| S_ij) - α²|hat(S)| hat(S)_ij )
        LM = jnp.zeros(grid.cell_shape())
        MM = jnp.zeros(grid.cell_shape())
        for i in range(ndim):
            for j in range(i, ndim):
                L_ij = (_box_filter_nd(u_cc[i] * u_cc[j], ndim, periodic_axes)
                        - u_hat[i] * u_hat[j])
                M_ij = (2.0 * delta ** 2
                        * (_box_filter_nd(S_mag * S[i, j], ndim, periodic_axes)
                           - alpha_sq * S_hat_mag * S_hat[i, j]))
                # Symmetric tensor: off-diagonal counted twice
                sym = 1.0 if i == j else 2.0
                LM = LM + sym * L_ij * M_ij
                MM = MM + sym * M_ij * M_ij

        # ── Lilly's least-squares Cs² ─────────────────────────────────
        Cs2 = jnp.clip(LM / (MM + 1e-12), 0.0, Cs_max ** 2)

        return Cs2 * delta ** 2 * S_mag

    return _dynamic_smagorinsky


# ── Neural Smagorinsky closure ───────────────────────────────────────────

def _init_neural_params(
    n_features: int,
    hidden_dim: int,
    n_layers: int,
    key,
) -> Dict[str, Any]:
    """Initialise MLP weights with Kaiming uniform init."""
    import jax.random as jr
    dims = [n_features] + [hidden_dim] * n_layers + [1]
    W_list, b_list = [], []
    n_total = len(dims) - 1
    for i in range(n_total):
        key, subkey = jr.split(key)
        scale = jnp.sqrt(2.0 / dims[i])
        W_list.append(jr.normal(subkey, (dims[i], dims[i + 1])) * scale)
        # Last layer: bias = 1.0 so softplus output starts ≈ 1.3 (avoids dead zone)
        if i == n_total - 1:
            b_list.append(jnp.ones(dims[i + 1]))
        else:
            b_list.append(jnp.zeros(dims[i + 1]))
    return {'W': W_list, 'b': b_list}


def _neural_forward(params: Dict[str, Any], features: jnp.ndarray) -> jnp.ndarray:
    """MLP forward: features (N, F) -> cs_pred (N, 1), guaranteed ≥ 0."""
    x = features
    n_layers = len(params['W'])
    for i in range(n_layers):
        x = x @ params['W'][i] + params['b'][i]
        if i < n_layers - 1:
            x = jax.nn.relu(x)
    return jax.nn.softplus(x)  # ensures Cs > 0


def make_neural_closure(
    hidden_dim: int = 32,
    n_layers: int = 2,
    params: Dict[str, Any] = None,
    key=None,
) -> Tuple[ViscosityClosureFn, Dict[str, Any]]:
    """Create a neural Smagorinsky closure.

    The network predicts a spatially-varying Smagorinsky coefficient
    ``Cs(x)`` from local flow features, then computes::

        ν_t(x) = (Cs(x) · Δ)² · |S(x)|

    This physics-constrained design guarantees non-negative eddy
    viscosity and correct dimensional scaling.

    Input features per cell (4):
        ``|S|``, ``|Ω|``, ``S²/(|S|² + ε)``, ``Δ``

    Parameters
    ----------
    hidden_dim : int
        Hidden layer width.
    n_layers : int
        Number of hidden layers.
    params : dict, optional
        Pre-trained weight dict ``{'W': [...], 'b': [...]}``.
        When ``None``, random weights are initialised (requires *key*).
    key : jax PRNGKey, optional
        Random key for weight initialisation.

    Returns
    -------
    closure_fn : ViscosityClosureFn
        ``closure_fn(state, grid, params_dict) -> nu_t``
        where ``params_dict['neural_params']`` holds the MLP weights.
    init_params : dict
        Initial (or provided) MLP weight dict.
    """
    import jax.random as jr
    n_features = 4

    if params is None:
        if key is None:
            key = jr.PRNGKey(0)
        init_params = _init_neural_params(n_features, hidden_dim, n_layers, key)
    else:
        init_params = params

    def _neural_smagorinsky(
        state: FluidState,
        grid: MACGrid,
        params_dict: Dict[str, Any],
    ) -> jnp.ndarray:
        ndim = grid.ndim
        nn_params = params_dict.get('neural_params', init_params)

        # Velocity gradient tensor
        g = velocity_gradient_tensor(state.velocity, grid)
        g_t = jnp.transpose(g, axes=(1, 0) + tuple(range(2, 2 + ndim)))

        # Strain rate S and vorticity Omega
        S = 0.5 * (g + g_t)
        Omega = 0.5 * (g - g_t)
        S_sq = jnp.einsum('ij...,ij...->...', S, S)
        O_sq = jnp.einsum('ij...,ij...->...', Omega, Omega)
        S_mag = jnp.sqrt(jnp.maximum(2.0 * S_sq, 0.0))
        O_mag = jnp.sqrt(jnp.maximum(2.0 * O_sq, 0.0))

        # Filter width
        delta = grid.cell_volumes() ** (1.0 / ndim)

        # Normalised features: [|S|, |Ω|, S²/(|S|²+ε), Δ]
        S_mag_safe = jnp.maximum(S_mag, 1e-10)
        features = jnp.stack([
            S_mag,
            O_mag,
            S_sq / (S_mag_safe ** 2 + 1e-10),
            jnp.broadcast_to(delta, S_mag.shape),
        ], axis=-1)  # (*cell_shape, 4)

        # Flatten cells → (N, 4)
        cell_shape = grid.cell_shape()
        n_cells = 1
        for s in cell_shape:
            n_cells *= s
        features_flat = features.reshape(n_cells, n_features)

        # MLP forward
        cs_pred = _neural_forward(nn_params, features_flat)  # (N, 1)
        cs_pred = cs_pred.reshape(cell_shape)

        # ν_t = (Cs · Δ)² · |S|
        return (cs_pred * delta) ** 2 * S_mag

    return _neural_smagorinsky, init_params
