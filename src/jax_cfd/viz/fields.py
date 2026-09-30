"""
Field visualization for JAX-CFD simulation outputs (2D and 3D).

All functions read directly from HDF5 files produced by H5Writer,
so they can be run after (or even during) a simulation without
re-running any physics.

Usage examples
--------------
# List available fields in a file:
from jax_cfd.viz.fields import list_fields
list_fields('output/run.h5')          # prints available field names

# Single frame (2D):
from jax_cfd.viz.fields import plot_field
plot_field('output/run.h5', step=2000, field='vorticity', clim=(-5, 5))
plot_field('output/run.h5', field='T')          # temperature (heat solver)

# 3D data — slice along z-axis at the midpoint:
plot_field('output/run.h5', field='vorticity')  # auto: z mid-plane
plot_field('output/run.h5', field='speed', slice_axis='z', slice_index=10)

# Multiple fields side by side:
from jax_cfd.viz.fields import plot_fields
plot_fields('output/run.h5', fields=['vorticity', 'T'])

# Animation saved to file:
from jax_cfd.viz.fields import animate
animate('output/run.h5', field='vorticity', clim=(-5, 5), save='vorticity.mp4')

Supported built-in field names
--------------------------------
'vorticity'  2D: ωz = dv/dx - du/dy;  3D: |omega| = |curl(u)|
'speed'      |u| = sqrt(u^2+v^2[+w^2])
'pressure'   p
'u'          x-velocity (cell-centred)
'v'          y-velocity (cell-centred)
'w'          z-velocity (cell-centred, 3D only)
'T'          temperature (stored; requires heat solver)
<any key>    any other scalar stored in the HDF5 snapshot
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from typing import Optional, Tuple, Union

from jax_cfd.io.reader import load_grid, load_snapshot, list_steps


# ── Derived field computation ─────────────────────────────────────────────

def compute_vorticity(
    u: np.ndarray,   # (Nx+1, Ny)
    v: np.ndarray,   # (Nx, Ny+1)
    dx: float,
    dy: float,
) -> np.ndarray:     # (Nx, Ny)
    """
    Cell-centred vorticity  wz = dv/dx - du/dy,  shape (Nx, Ny).

    Each cell value is the average curl over the four corners:
        wz[i,j] = ( v[i+1,j]+v[i+1,j+1] - v[i,j]-v[i,j+1] ) / (2 dx)
                - ( u[i,j+1]+u[i+1,j+1] - u[i,j]-u[i+1,j] ) / (2 dy)
    """
    v_pad = np.pad(v, ((0, 1), (0, 0)), mode='edge')
    u_pad = np.pad(u, ((0, 0), (0, 1)), mode='edge')

    dvdx = (v_pad[1:, :-1] + v_pad[1:, 1:] - v_pad[:-1, :-1] - v_pad[:-1, 1:]) / (2.0 * dx)
    dudy = (u_pad[:-1, 1:] + u_pad[1:, 1:] - u_pad[:-1, :-1] - u_pad[1:, :-1]) / (2.0 * dy)
    return dvdx - dudy


def compute_vorticity_3d(
    u: np.ndarray,    # (Nx+1, Ny, Nz)
    v: np.ndarray,    # (Nx, Ny+1, Nz)
    w: np.ndarray,    # (Nx, Ny, Nz+1)
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:      # (Nx, Ny, Nz)
    """
    Cell-centred vorticity magnitude  |omega| = sqrt(wx^2+wy^2+wz^2).

    Components: wx = dw/dy - dv/dz,  wy = du/dz - dw/dx,  wz = dv/dx - du/dy.
    Uses cell-centred velocities with central differences.
    """
    # Face-to-cell average
    u_cc = 0.5 * (u[:-1, :, :] + u[1:, :, :])   # (Nx, Ny, Nz)
    v_cc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])   # (Nx, Ny, Nz)
    w_cc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])   # (Nx, Ny, Nz)

    def _dd(f, axis, h):
        """Central difference with edge padding."""
        pw = [(0, 0)] * 3
        pw[axis] = (1, 1)
        fp = np.pad(f, pw, mode='edge')
        sl_p = [slice(None)] * 3
        sl_m = [slice(None)] * 3
        sl_p[axis] = slice(2, None)
        sl_m[axis] = slice(None, -2)
        return (fp[tuple(sl_p)] - fp[tuple(sl_m)]) / (2.0 * h)

    # omega = curl(u)
    wx = _dd(w_cc, 1, dy) - _dd(v_cc, 2, dz)
    wy = _dd(u_cc, 2, dz) - _dd(w_cc, 0, dx)
    wz = _dd(v_cc, 0, dx) - _dd(u_cc, 1, dy)

    return np.sqrt(wx**2 + wy**2 + wz**2)


def compute_speed(
    u: np.ndarray,   # (Nx+1, Ny)
    v: np.ndarray,   # (Nx, Ny+1)
) -> np.ndarray:     # (Nx, Ny)
    """Cell-centred velocity magnitude |u| (2D)."""
    u_cc = 0.5 * (u[:-1, :] + u[1:, :])
    v_cc = 0.5 * (v[:, :-1] + v[:, 1:])
    return np.sqrt(u_cc**2 + v_cc**2)


def compute_speed_3d(
    u: np.ndarray,    # (Nx+1, Ny, Nz)
    v: np.ndarray,    # (Nx, Ny+1, Nz)
    w: np.ndarray,    # (Nx, Ny, Nz+1)
) -> np.ndarray:      # (Nx, Ny, Nz)
    """Cell-centred velocity magnitude |u| (3D)."""
    u_cc = 0.5 * (u[:-1, :, :] + u[1:, :, :])
    v_cc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    w_cc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    return np.sqrt(u_cc**2 + v_cc**2 + w_cc**2)


# ── Helpers ───────────────────────────────────────────────────────────────

def _detect_ndim(grid: dict) -> int:
    """Detect dimensionality from grid coordinate arrays."""
    return 3 if 'zc' in grid else 2


def _face_to_cell_nd(f: np.ndarray, axis: int) -> np.ndarray:
    """Average a face-centred array to cell centres along *axis*."""
    sl_p = [slice(None)] * f.ndim
    sl_m = [slice(None)] * f.ndim
    sl_p[axis] = slice(1, None)
    sl_m[axis] = slice(None, -1)
    return 0.5 * (f[tuple(sl_p)] + f[tuple(sl_m)])


def _take_slice(data_3d: np.ndarray, slice_axis: int, slice_index: int) -> np.ndarray:
    """Extract a 2D slice from a 3D array."""
    sl = [slice(None)] * 3
    sl[slice_axis] = slice_index
    return data_3d[tuple(sl)]


_SLICE_AXIS_MAP = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}


def _resolve_slice(
    grid: dict,
    slice_axis: Optional[Union[str, int]],
    slice_index: Optional[int],
) -> Tuple[int, int]:
    """Return (axis_int, index_int) with defaults for 3D data."""
    if slice_axis is None:
        slice_axis = 2   # default: z-plane
    axis = _SLICE_AXIS_MAP.get(slice_axis, slice_axis)
    if not isinstance(axis, int) or axis not in (0, 1, 2):
        raise ValueError(f"slice_axis must be 'x', 'y', 'z', 0, 1, or 2, got {slice_axis}")
    coord_key = ['xc', 'yc', 'zc'][axis]
    n = len(grid[coord_key])
    if slice_index is None:
        slice_index = n // 2
    if slice_index < 0 or slice_index >= n:
        raise ValueError(f"slice_index={slice_index} out of range [0, {n-1}] for axis {axis}")
    return axis, slice_index


def _get_plot_axes(grid: dict, slice_axis: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Return (x_coords, y_coords, xlabel, ylabel) for a 2D plot of a 3D slice."""
    names = ['x', 'y', 'z']
    keys = ['xc', 'yc', 'zc']
    plot_dims = [d for d in range(3) if d != slice_axis]
    return (
        grid[keys[plot_dims[0]]],
        grid[keys[plot_dims[1]]],
        names[plot_dims[0]],
        names[plot_dims[1]],
    )


# ── Field selector ─────────────────────────────────────────────────────────

# Default (cmap, symmetric_clim) for known fields.
_FIELD_DEFAULTS: dict = {
    'vorticity': ('RdBu_r',  True),
    'speed':     ('viridis', False),
    'pressure':  ('RdBu_r',  True),
    'u':         ('RdBu_r',  True),
    'v':         ('RdBu_r',  True),
    'w':         ('RdBu_r',  True),
    'T':         ('RdBu_r',  False),
}


def list_fields(h5_path: str, step: Optional[int] = None) -> None:
    """Print available field names for the given HDF5 file."""
    snap = load_snapshot(h5_path, step)
    stored = [k for k in snap if k not in ('t', 'step')]
    derived = ['vorticity', 'speed']
    extra = [k for k in stored if k not in ('u', 'v', 'w', 'p')]

    grid = load_grid(h5_path)
    ndim = _detect_ndim(grid)

    print(f"\nAvailable fields in '{h5_path}'  ({ndim}D):")
    print(f"  Derived  : {', '.join(derived)}")
    print(f"  Stored   : {', '.join(stored)}")
    if extra:
        print(f"  Scalars  : {', '.join(extra)}  (e.g., temperature 'T')")
    if ndim == 3:
        print(f"  (3D data — use slice_axis='x'/'y'/'z' and slice_index to select a plane)")
    print()


def _get_field_full(snap: dict, grid: dict, field: str) -> Tuple[np.ndarray, str, str]:
    """
    Return (data_nd, colormap, label) for the requested field name.

    For 2D data returns (Nx, Ny), for 3D returns (Nx, Ny, Nz).
    """
    ndim = _detect_ndim(grid)
    dx = float(grid['xc'][1] - grid['xc'][0])
    dy = float(grid['yc'][1] - grid['yc'][0])

    if field == 'vorticity':
        if ndim == 2:
            data = compute_vorticity(snap['u'], snap['v'], dx, dy)
            label = 'wz'
        else:
            dz = float(grid['zc'][1] - grid['zc'][0])
            data = compute_vorticity_3d(snap['u'], snap['v'], snap['w'], dx, dy, dz)
            label = '|omega|'
        cmap = 'RdBu_r' if ndim == 2 else 'inferno'
    elif field == 'speed':
        if ndim == 2:
            data = compute_speed(snap['u'], snap['v'])
        else:
            data = compute_speed_3d(snap['u'], snap['v'], snap['w'])
        cmap, label = 'viridis', '|u|'
    elif field == 'pressure':
        data = snap['p']
        cmap, label = 'RdBu_r', 'p'
    elif field in ('u', 'v', 'w'):
        idx = {'u': 0, 'v': 1, 'w': 2}[field]
        if field == 'w' and ndim == 2:
            raise ValueError("'w' field only available for 3D simulations.")
        comp_names = ['u', 'v', 'w'][:ndim]
        data = _face_to_cell_nd(snap[comp_names[idx]], idx)
        cmap, label = 'RdBu_r', field
    elif field in snap:
        data = np.array(snap[field])
        cmap = _FIELD_DEFAULTS.get(field, ('viridis', False))[0]
        label = field
    else:
        stored = [k for k in snap if k not in ('t', 'step')]
        available = ['vorticity', 'speed'] + stored
        raise ValueError(
            f"Unknown field '{field}'.  Available: {available}"
        )
    return data, cmap, label


# ── Single frame ───────────────────────────────────────────────────────────

def plot_field(
    h5_path: str,
    step: Optional[int] = None,
    field: str = 'vorticity',
    clim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (12, 5),
    save: Optional[str] = None,
    show: bool = True,
    slice_axis: Optional[Union[str, int]] = None,
    slice_index: Optional[int] = None,
) -> plt.Figure:
    """
    Plot a single field snapshot.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 run file.
    step : int, optional
        Step to plot.  Defaults to the last saved step.
    field : str
        One of: 'vorticity', 'speed', 'pressure', 'u', 'v', 'w', 'T', ...
    clim : (vmin, vmax), optional
        Colour axis limits.  Auto-scaled if None.
    save : str, optional
        If given, save the figure to this path instead of displaying.
    slice_axis : str or int, optional
        For 3D data: 'x'/'y'/'z' (or 0/1/2).  Default: 'z'.
    slice_index : int, optional
        Index along slice_axis.  Default: midpoint.
    """
    grid = load_grid(h5_path)
    snap = load_snapshot(h5_path, step)
    ndim = _detect_ndim(grid)

    data, cmap, label = _get_field_full(snap, grid, field)

    if ndim == 3:
        ax_int, idx = _resolve_slice(grid, slice_axis, slice_index)
        data_2d = _take_slice(data, ax_int, idx)
        xc, yc, xlabel, ylabel = _get_plot_axes(grid, ax_int)
        ax_name = ['x', 'y', 'z'][ax_int]
        coord_key = ['xc', 'yc', 'zc'][ax_int]
        slice_val = float(grid[coord_key][idx])
        slice_info = f'{ax_name}={slice_val:.3f} (index {idx})'
    else:
        data_2d = data
        xc, yc = grid['xc'], grid['yc']
        xlabel, ylabel = 'x', 'y'
        slice_info = None

    fig, ax = plt.subplots(figsize=figsize)
    t_str = f't = {snap["t"]:.2f},  step = {snap["step"]}'
    if slice_info:
        t_str += f',  slice {slice_info}'

    vmin, vmax = clim if clim else (None, None)
    if clim is None:
        _, symmetric = _FIELD_DEFAULTS.get(field, ('viridis', False))
        if ndim == 3 and field == 'vorticity':
            symmetric = False  # magnitude is non-negative
        if symmetric:
            amax = np.percentile(np.abs(data_2d), 99)
            vmin, vmax = -amax, amax

    im = ax.pcolormesh(xc, yc, data_2d.T, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='auto', rasterized=True)
    plt.colorbar(im, ax=ax, label=label, shrink=0.8)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f'{field}   ({t_str})')

    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    elif show:
        plt.show()
    return fig


# ── Multiple fields ────────────────────────────────────────────────────────

def plot_fields(
    h5_path: str,
    fields: Tuple[str, ...] = ('vorticity', 'pressure'),
    step: Optional[int] = None,
    clims: Optional[dict] = None,
    figsize_per_panel: Tuple[float, float] = (6, 4),
    save: Optional[str] = None,
    show: bool = True,
    slice_axis: Optional[Union[str, int]] = None,
    slice_index: Optional[int] = None,
) -> plt.Figure:
    """
    Plot several fields side by side in a single figure.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 run file.
    fields : tuple of str
        Field names to plot, in order.
    step : int or None
        Snapshot step.  None -> last saved step.
    clims : dict or None
        Per-field colour limits, e.g. {'vorticity': (-5, 5), 'T': (0, 1)}.
    slice_axis, slice_index : optional
        For 3D data, select the slicing plane.
    """
    grid = load_grid(h5_path)
    snap = load_snapshot(h5_path, step)
    ndim = _detect_ndim(grid)

    if ndim == 3:
        ax_int, idx = _resolve_slice(grid, slice_axis, slice_index)
        xc, yc, xlabel, ylabel = _get_plot_axes(grid, ax_int)
    else:
        xc, yc = grid['xc'], grid['yc']
        xlabel, ylabel = 'x', 'y'
        ax_int, idx = None, None

    t_str = f't = {snap["t"]:.2f},  step = {snap["step"]}'

    n = len(fields)
    fw, fh = figsize_per_panel
    fig, axes = plt.subplots(1, n, figsize=(fw * n, fh))
    if n == 1:
        axes = [axes]

    clims = clims or {}

    for ax, field_name in zip(axes, fields):
        data, cmap, label = _get_field_full(snap, grid, field_name)
        if ndim == 3:
            data = _take_slice(data, ax_int, idx)

        clim = clims.get(field_name)
        vmin, vmax = clim if clim else (None, None)
        if clim is None:
            _, symmetric = _FIELD_DEFAULTS.get(field_name, ('viridis', False))
            if ndim == 3 and field_name == 'vorticity':
                symmetric = False
            if symmetric:
                amax = np.percentile(np.abs(data), 99)
                vmin, vmax = -amax, amax

        im = ax.pcolormesh(xc, yc, data.T, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           shading='auto', rasterized=True)
        plt.colorbar(im, ax=ax, label=label, shrink=0.8)
        ax.set_aspect('equal')
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(field_name)

    fig.suptitle(t_str, fontsize=11)
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    elif show:
        plt.show()
    return fig


# ── Animation ──────────────────────────────────────────────────────────────

def animate(
    h5_path: str,
    field: str = 'vorticity',
    clim: Optional[Tuple[float, float]] = None,
    fps: int = 10,
    figsize: Tuple[float, float] = (12, 5),
    save: Optional[str] = None,
    show: bool = True,
    slice_axis: Optional[Union[str, int]] = None,
    slice_index: Optional[int] = None,
) -> manim.FuncAnimation:
    """
    Animate all saved snapshots.

    Parameters
    ----------
    save : str, optional
        If given, save to this path.  Extension determines format:
        ``.mp4`` (requires ffmpeg), ``.gif`` (requires pillow).
    slice_axis, slice_index : optional
        For 3D data, select the slicing plane (default: z midpoint).
    """
    grid = load_grid(h5_path)
    steps = list_steps(h5_path)
    ndim = _detect_ndim(grid)
    if not steps:
        raise ValueError(f"No snapshots found in {h5_path}")

    if ndim == 3:
        ax_int, idx = _resolve_slice(grid, slice_axis, slice_index)
        xc, yc, xlabel, ylabel = _get_plot_axes(grid, ax_int)
    else:
        xc, yc = grid['xc'], grid['yc']
        xlabel, ylabel = 'x', 'y'
        ax_int, idx = None, None

    def _get_2d(snap_):
        data_, cmap_, label_ = _get_field_full(snap_, grid, field)
        if ndim == 3:
            data_ = _take_slice(data_, ax_int, idx)
        return data_, cmap_, label_

    # Pre-compute colour limits from the first snapshot if not given
    if clim is None:
        snap0 = load_snapshot(h5_path, steps[0])
        data0, _, _ = _get_2d(snap0)
        amax = np.percentile(np.abs(data0), 99) or 1.0
        _, symmetric = _FIELD_DEFAULTS.get(field, ('viridis', False))
        if ndim == 3 and field == 'vorticity':
            symmetric = False
        if symmetric:
            clim = (-amax, amax)
        else:
            clim = (float(data0.min()), float(data0.max()) or 1.0)

    _, cmap, label = _get_2d(load_snapshot(h5_path, steps[0]))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    snap0 = load_snapshot(h5_path, steps[0])
    d0, _, _ = _get_2d(snap0)
    im = ax.pcolormesh(xc, yc, d0.T, cmap=cmap,
                       vmin=clim[0], vmax=clim[1],
                       shading='auto', rasterized=True)
    plt.colorbar(im, ax=ax, label=label, shrink=0.8)
    title = ax.set_title('')

    def update(step):
        snap = load_snapshot(h5_path, step)
        data, _, _ = _get_2d(snap)
        im.set_array(data.T.ravel())
        title.set_text(f'{field}   t = {snap["t"]:.2f},  step = {snap["step"]}')
        return im, title

    ani = manim.FuncAnimation(fig, update, frames=steps,
                               interval=1000 // fps, blit=True)

    if save:
        writer_cls = 'ffmpeg' if save.endswith('.mp4') else 'pillow'
        ani.save(save, writer=writer_cls, fps=fps, dpi=100)
    elif show:
        plt.show()
    return ani
