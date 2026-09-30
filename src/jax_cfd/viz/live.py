"""
Real-time visualization callbacks for JAX-CFD simulations.

These classes are designed to be passed as ``callbacks`` to
:func:`jax_cfd.runner.run`.  They use matplotlib's interactive mode
(``plt.ion()``) for non-blocking updates.

Usage
-----
::

    from jax_cfd.viz.live import LiveDashboard, LiveFieldViewer

    # Monitor scalar quantities in real time
    dash = LiveDashboard(quantities=['Cd', 'Cl', 'div'], window=500)

    # Watch the vorticity field evolve
    viewer = LiveFieldViewer(field='vorticity', clim=(-5, 5), update_every=10)

    from jax_cfd.runner import run
    run(CONFIG, callbacks=[dash, viewer])
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple


# ── LiveDashboard ─────────────────────────────────────────────────────────

class LiveDashboard:
    """
    Real-time line plots of scalar monitor quantities (Cd, Cl, div, ke, …).

    Parameters
    ----------
    quantities : list of str
        Monitor keys to plot.  Each gets its own subplot row.
    window : int
        Maximum number of data points shown (sliding window).
    figsize : tuple
        Figure size in inches (width, height).
    """

    def __init__(
        self,
        quantities: Sequence[str] = ('Cd', 'Cl'),
        window: int = 500,
        figsize: Tuple[float, float] = (8, 5),
    ):
        self.quantities = list(quantities)
        self.window = window
        self._t_buf: list = []
        self._data: dict = {q: [] for q in self.quantities}
        self._fig = None
        self._axes = None
        self._lines = None

    def _init_figure(self):
        nq = len(self.quantities)
        plt.ion()
        self._fig, axes = plt.subplots(nq, 1, figsize=(8, 2.0 * nq),
                                       sharex=True, squeeze=False)
        self._axes = [ax for ax in axes[:, 0]]
        self._lines = []
        for ax, q in zip(self._axes, self.quantities):
            line, = ax.plot([], [], 'b-', linewidth=0.8)
            ax.set_ylabel(q)
            ax.grid(True, alpha=0.3)
            self._lines.append(line)
        self._axes[-1].set_xlabel('t')
        self._fig.suptitle('Live Monitor', fontsize=10)
        self._fig.tight_layout()

    def __call__(self, *, step, t, state, monitor, grid, cfl, dt, **kw):
        if self._fig is None:
            self._init_figure()

        self._t_buf.append(t)
        for q in self.quantities:
            self._data[q].append(monitor.get(q, float('nan')))

        # Sliding window
        if len(self._t_buf) > self.window:
            self._t_buf = self._t_buf[-self.window:]
            for q in self.quantities:
                self._data[q] = self._data[q][-self.window:]

        t_arr = np.array(self._t_buf)
        for line, ax, q in zip(self._lines, self._axes, self.quantities):
            y = np.array(self._data[q])
            line.set_data(t_arr, y)
            ax.set_xlim(t_arr[0], t_arr[-1])
            finite = y[np.isfinite(y)]
            if len(finite) > 0:
                ymin, ymax = finite.min(), finite.max()
                margin = max(0.05 * (ymax - ymin), 1e-8)
                ax.set_ylim(ymin - margin, ymax + margin)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None


# ── LiveFieldViewer ───────────────────────────────────────────────────────

class LiveFieldViewer:
    """
    Real-time 2-D field visualization (vorticity, speed, pressure, T, …).

    Parameters
    ----------
    field : str
        Field to display.  Built-in derived fields: ``'vorticity'``,
        ``'speed'``.  Or any scalar name stored in ``state.scalars``.
    clim : tuple of float or None
        Colour limits ``(vmin, vmax)``.  ``None`` → auto-scale each frame.
    cmap : str
        Matplotlib colourmap name.
    update_every : int
        Update the plot every *N* callback invocations (default 1 = every
        ``diag_every`` steps).  Increase to reduce overhead.
    figsize : tuple
        Figure size in inches.
    """

    _DEFAULTS = {
        'vorticity': ('RdBu_r',  True),
        'speed':     ('viridis', False),
        'pressure':  ('RdBu_r',  True),
        'u':         ('RdBu_r',  True),
        'v':         ('RdBu_r',  True),
        'T':         ('inferno', False),
    }

    def __init__(
        self,
        field: str = 'vorticity',
        clim: Optional[Tuple[float, float]] = None,
        cmap: Optional[str] = None,
        update_every: int = 1,
        figsize: Tuple[float, float] = (8, 5),
    ):
        self.field = field
        self.clim = clim
        self.update_every = max(1, update_every)
        self.figsize = figsize
        self._call_count = 0

        default_cmap, self._symmetric = self._DEFAULTS.get(field, ('viridis', False))
        self.cmap = cmap or default_cmap

        self._fig = None
        self._ax = None
        self._im = None

    def _compute_field(self, state, grid):
        """Extract a 2-D numpy array from the current FluidState."""
        vel = state.velocity
        ndim = grid.ndim

        if self.field == 'vorticity':
            if ndim == 2:
                u = np.asarray(vel[0])  # (Nx+1, Ny)
                v = np.asarray(vel[1])  # (Nx, Ny+1)
                dx, dy = grid.spacings
                u_cc = 0.5 * (u[:-1, :] + u[1:, :])
                v_cc = 0.5 * (v[:, :-1] + v[:, 1:])
                v_pad = np.pad(v_cc, ((1, 1), (0, 0)), mode='edge')
                u_pad = np.pad(u_cc, ((0, 0), (1, 1)), mode='edge')
                dv_dx = (v_pad[2:, :] - v_pad[:-2, :]) / (2.0 * dx)
                du_dy = (u_pad[:, 2:] - u_pad[:, :-2]) / (2.0 * dy)
                return dv_dx - du_dy
            else:
                # 3D: show mid-z slice of omega_z
                u = np.asarray(vel[0])
                v = np.asarray(vel[1])
                dx, dy = grid.spacings[0], grid.spacings[1]
                k_mid = grid.shape[2] // 2
                u2d = u[:, :, k_mid]
                v2d = v[:, :, k_mid]
                u_cc = 0.5 * (u2d[:-1, :] + u2d[1:, :])
                v_cc = 0.5 * (v2d[:, :-1] + v2d[:, 1:])
                v_pad = np.pad(v_cc, ((1, 1), (0, 0)), mode='edge')
                u_pad = np.pad(u_cc, ((0, 0), (1, 1)), mode='edge')
                dv_dx = (v_pad[2:, :] - v_pad[:-2, :]) / (2.0 * dx)
                du_dy = (u_pad[:, 2:] - u_pad[:, :-2]) / (2.0 * dy)
                return dv_dx - du_dy

        elif self.field == 'speed':
            if ndim == 2:
                u = np.asarray(vel[0])
                v = np.asarray(vel[1])
                u_cc = 0.5 * (u[:-1, :] + u[1:, :])
                v_cc = 0.5 * (v[:, :-1] + v[:, 1:])
                return np.sqrt(u_cc**2 + v_cc**2)
            else:
                u = np.asarray(vel[0])
                v = np.asarray(vel[1])
                w = np.asarray(vel[2])
                k_mid = grid.shape[2] // 2
                u_cc = 0.5 * (u[:-1, :, k_mid] + u[1:, :, k_mid])
                v_cc = 0.5 * (v[:, :-1, k_mid] + v[:, 1:, k_mid])
                w_cc = 0.5 * (w[:, :, k_mid] + w[:, :, k_mid + 1])
                return np.sqrt(u_cc**2 + v_cc**2 + w_cc**2)

        elif self.field == 'pressure':
            p = np.asarray(state.pressure)
            if ndim == 3:
                p = p[:, :, grid.shape[2] // 2]
            return p

        elif self.field in ('u', 'v', 'w'):
            idx = {'u': 0, 'v': 1, 'w': 2}[self.field]
            f = np.asarray(vel[idx])
            axis = idx
            sl_p = [slice(None)] * ndim
            sl_m = [slice(None)] * ndim
            sl_p[axis] = slice(1, None)
            sl_m[axis] = slice(None, -1)
            f_cc = 0.5 * (f[tuple(sl_p)] + f[tuple(sl_m)])
            if ndim == 3:
                f_cc = f_cc[:, :, grid.shape[2] // 2]
            return f_cc

        else:
            # Try scalar fields (T, etc.)
            if state.scalars and self.field in state.scalars:
                s = np.asarray(state.scalars[self.field])
                if ndim == 3:
                    s = s[:, :, grid.shape[2] // 2]
                return s
            raise ValueError(f"Unknown field '{self.field}'")

    def _init_figure(self, data, grid):
        plt.ion()
        self._fig, self._ax = plt.subplots(1, 1, figsize=self.figsize)

        Lx, Ly = grid.domain_size[0], grid.domain_size[1]
        extent = [0, Lx, 0, Ly]

        vmin, vmax = self._get_clim(data)
        self._im = self._ax.imshow(
            data.T, origin='lower', aspect='equal',
            extent=extent, cmap=self.cmap, vmin=vmin, vmax=vmax,
            interpolation='bilinear',
        )
        self._fig.colorbar(self._im, ax=self._ax, shrink=0.8)
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')
        self._ax.set_title(f'{self.field}  t=0.000')
        self._fig.tight_layout()

    def _get_clim(self, data):
        if self.clim is not None:
            return self.clim
        if self._symmetric:
            vmax = max(abs(data.min()), abs(data.max()), 1e-10)
            return (-vmax, vmax)
        return (data.min(), data.max())

    def __call__(self, *, step, t, state, monitor, grid, cfl, dt, **kw):
        self._call_count += 1
        if self._call_count % self.update_every != 0:
            return

        data = self._compute_field(state, grid)

        if self._fig is None:
            self._init_figure(data, grid)
        else:
            self._im.set_data(data.T)
            if self.clim is None:
                vmin, vmax = self._get_clim(data)
                self._im.set_clim(vmin, vmax)
            self._ax.set_title(f'{self.field}  t={t:.3f}')

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
