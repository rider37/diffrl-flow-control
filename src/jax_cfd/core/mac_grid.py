from dataclasses import dataclass, field
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union

@dataclass(frozen=True)
class MACGrid:
    """
    N-Dimensional Staggered (Harlow-Welch MAC) Grid description.

    In a MAC grid:
    - Pressure (and scalars) are stored at cell centers: shape (N_1, N_2, ..., N_D)
    - i-th velocity component is stored at i-faces: shape is cell centers BUT the i-th dimension has N_i + 1 elements.

    Non-uniform (stretched) grids
    -----------------------------
    Set ``stretch={'axis': 1, 'type': 'tanh', 'ratio': 5.0}`` to cluster
    cells near both walls of the specified axis.  All existing code that uses
    ``grid.spacings`` (uniform tuple) continues to work unchanged; the new
    ``spacings_array(axis)`` method returns the per-cell 1-D spacing array
    when the axis is stretched.
    """
    shape: Tuple[int, ...]         # E.g., (Nx, Ny) or (Nx, Ny, Nz)
    domain_size: Tuple[float, ...] # E.g., (Lx, Ly) or (Lx, Ly, Lz)
    stretch: Optional[Dict[str, Any]] = field(default=None, hash=False,
                                              compare=False)

    def __post_init__(self):
        assert len(self.shape) == len(self.domain_size), \
            "Shape and size must be same dimensions."
        # Pre-compute and cache the stretched face coordinates (if any)
        if self.stretch is not None:
            fc = _compute_face_coords(self.stretch, self.shape, self.domain_size)
            object.__setattr__(self, '_stretched_faces', fc)
        else:
            object.__setattr__(self, '_stretched_faces', None)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def is_uniform(self) -> bool:
        return self.stretch is None

    @property
    def spacings(self) -> Tuple[float, ...]:
        """Uniform spacings (L/N per axis).  Always available for backward compat."""
        return tuple(L / N for L, N in zip(self.domain_size, self.shape))

    def spacings_array(self, axis: int) -> Union[float, jnp.ndarray]:
        """
        Per-cell spacing along *axis*.

        Returns a **scalar** ``float`` for uniform axes (identical to
        ``self.spacings[axis]``) and a **1-D jnp.ndarray** of length
        ``shape[axis]`` for the stretched axis.
        """
        if self.stretch is None or self.stretch.get('axis') != axis:
            return self.spacings[axis]
        fc = self._stretched_faces  # type: ignore[attr-defined]
        return jnp.diff(fc)

    def face_coords_1d(self, axis: int) -> jnp.ndarray:
        """
        1-D array of face positions along *axis*, length ``shape[axis]+1``.

        For uniform axes this is ``linspace(0, L, N+1)``.
        For the stretched axis it returns the non-uniform face coordinates.
        """
        if self.stretch is not None and self.stretch.get('axis') == axis:
            return self._stretched_faces  # type: ignore[attr-defined]
        L = self.domain_size[axis]
        N = self.shape[axis]
        return jnp.linspace(0.0, L, N + 1)

    def cell_centers_1d(self, axis: int) -> jnp.ndarray:
        """
        1-D array of cell-centre positions along *axis*, length ``shape[axis]``.
        """
        fc = self.face_coords_1d(axis)
        return 0.5 * (fc[:-1] + fc[1:])

    def cell_volumes(self) -> jnp.ndarray:
        """
        Per-cell volume array, shape ``cell_shape()``.

        For fully uniform grids this is a constant array (but still the
        correct shape for broadcasting).
        """
        ndim = self.ndim
        vol = jnp.ones(self.cell_shape())
        for ax in range(ndim):
            dx_ax = self.spacings_array(ax)
            if isinstance(dx_ax, (float, int)):
                vol = vol * dx_ax
            else:
                shape_bc = [1] * ndim
                shape_bc[ax] = self.shape[ax]
                vol = vol * jnp.reshape(dx_ax, shape_bc)
        return vol

    def min_spacing(self) -> float:
        """Minimum cell spacing across all axes (for CFL)."""
        mins = []
        for ax in range(self.ndim):
            dx_ax = self.spacings_array(ax)
            if isinstance(dx_ax, (float, int)):
                mins.append(dx_ax)
            else:
                mins.append(float(jnp.min(dx_ax)))
        return min(mins)

    # ── Legacy interface (unchanged) ──────────────────────────────────────

    def cell_shape(self) -> Tuple[int, ...]:
        return self.shape

    def face_shape(self, axis: int) -> Tuple[int, ...]:
        """Returns the tensor shape for the velocity component along `axis`."""
        face_shp = list(self.shape)
        face_shp[axis] += 1
        return tuple(face_shp)

    def cell_centers(self) -> Tuple[jnp.ndarray, ...]:
        """1D arrays of cell-centre positions along each axis."""
        return tuple(self.cell_centers_1d(ax) for ax in range(self.ndim))

    def face_centers(self, axis: int) -> Tuple[jnp.ndarray, ...]:
        """1D arrays of positions for the `axis`-faces."""
        coords = list(self.cell_centers())
        coords[axis] = self.face_coords_1d(axis)
        return tuple(coords)

    def cell_mesh(self) -> Tuple[jnp.ndarray, ...]:
        """Returns N-D meshgrid arrays (X, Y, [Z]) at cell centres."""
        return jnp.meshgrid(*self.cell_centers(), indexing='ij')

    def face_mesh(self, axis: int) -> Tuple[jnp.ndarray, ...]:
        """Returns N-D meshgrid arrays exactly on the staggered faces along `axis`."""
        return jnp.meshgrid(*self.face_centers(axis), indexing='ij')


# ── Stretching functions ──────────────────────────────────────────────────

def _compute_face_coords(
    stretch_cfg: Dict[str, Any],
    shape: Tuple[int, ...],
    domain_size: Tuple[float, ...],
) -> jnp.ndarray:
    """Dispatch to the appropriate stretching function."""
    stype = stretch_cfg.get('type', 'tanh')
    axis = stretch_cfg['axis']
    N = shape[axis]
    L = domain_size[axis]
    ratio = stretch_cfg.get('ratio', 5.0)

    if stype == 'tanh':
        return tanh_stretch(N, L, ratio)
    elif stype == 'tanh_center':
        return tanh_center_stretch(N, L, ratio)
    else:
        raise ValueError(f"Unknown stretch type '{stype}'. Supported: tanh, tanh_center")


def tanh_stretch(N: int, L: float, ratio: float) -> jnp.ndarray:
    """
    Two-sided hyperbolic-tangent stretching (both walls clustered).

    Parameters
    ----------
    N : int
        Number of cells.
    L : float
        Domain length.
    ratio : float
        Stretching ratio ``dy_max / dy_min``.  ratio=1 → uniform.

    Returns
    -------
    face_coords : jnp.ndarray, shape (N+1,)
        Monotonically increasing face coordinates in [0, L].

    Notes
    -----
    The mapping is:

        y(η) = L/2 · (1 + tanh(δ·(2η − 1)) / tanh(δ))

    where η ∈ [0, 1] is the uniform computational coordinate and δ is
    chosen so that ``dy_max / dy_min = ratio``.
    """
    if ratio <= 1.0 + 1e-8:
        return jnp.linspace(0.0, L, N + 1)

    eta = np.linspace(0.0, 1.0, N + 1)

    def _discrete_ratio(delta):
        """Actual dy_max/dy_min for the discrete N-cell grid."""
        y = 0.5 * (1.0 + np.tanh(delta * (2.0 * eta - 1.0)) / np.tanh(delta))
        dy = np.diff(y)
        return dy.max() / dy.min()

    # Bisection on delta to match the target ratio exactly for this N.
    # Continuous limit: ratio = cosh²(delta), so start near arccosh(√ratio).
    lo, hi = 0.01, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _discrete_ratio(mid) < ratio:
            lo = mid
        else:
            hi = mid
    delta = 0.5 * (lo + hi)

    y = L * 0.5 * (1.0 + np.tanh(delta * (2.0 * eta - 1.0)) / np.tanh(delta))
    y[0] = 0.0
    y[-1] = L
    return jnp.array(y)


def tanh_center_stretch(N: int, L: float, ratio: float) -> jnp.ndarray:
    """
    Center-clustered hyperbolic-tangent stretching.

    Cells are finest at the domain center (y = L/2) and coarsest at the
    boundaries.  This is the inverse mapping of ``tanh_stretch``.

    Parameters
    ----------
    N : int
        Number of cells.
    L : float
        Domain length.
    ratio : float
        Stretching ratio ``dy_max / dy_min``.  ratio=1 → uniform.

    Returns
    -------
    face_coords : jnp.ndarray, shape (N+1,)
        Monotonically increasing face coordinates in [0, L].
    """
    if ratio <= 1.0 + 1e-8:
        return jnp.linspace(0.0, L, N + 1)

    eta = np.linspace(0.0, 1.0, N + 1)

    def _discrete_ratio(delta):
        xi = np.tanh(delta) * (2.0 * eta - 1.0)
        xi = np.clip(xi, -1 + 1e-14, 1 - 1e-14)
        y = 0.5 * (1.0 + np.arctanh(xi) / delta)
        dy = np.diff(y)
        return dy.max() / dy.min()

    lo, hi = 0.01, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _discrete_ratio(mid) < ratio:
            lo = mid
        else:
            hi = mid
    delta = 0.5 * (lo + hi)

    xi = np.tanh(delta) * (2.0 * eta - 1.0)
    xi = np.clip(xi, -1 + 1e-14, 1 - 1e-14)
    y = L * 0.5 * (1.0 + np.arctanh(xi) / delta)
    y[0] = 0.0
    y[-1] = L
    return jnp.array(y)
