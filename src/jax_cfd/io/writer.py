"""
HDF5 Writer for JAX-CFD simulations (2D and 3D).

File layout
-----------
run.h5
├── attrs           config_json (str), created_at (str)
├── grid/           coordinate arrays (written once)
│   ├── attrs  ndim (int)
│   ├── xc  (Nx,)   cell-centre x
│   ├── yc  (Ny,)   cell-centre y
│   ├── [zc (Nz,)]  cell-centre z  [3D only]
│   ├── xu  (Nx+1,) u-face x
│   ├── yv  (Ny+1,) v-face y
│   └── [zw (Nz+1,)] w-face z      [3D only]
├── fields/         field snapshots (one group per save)
│   └── step_0000200/
│       ├── attrs   t (float), step (int)
│       ├── u  (Nx+1, Ny[, Nz])
│       ├── v  (Nx, Ny+1[, Nz])
│       ├── [w (Nx, Ny, Nz+1)]     [3D only]
│       ├── p  (Nx, Ny[, Nz])
│       └── T  (Nx, Ny[, Nz])      [heat solver only]
├── diagnostics/    IBM force time-series (resizable)
│   ├── step  (N,)
│   ├── t     (N,)
│   ├── Fx, Fy [, Fz]  (N,)
│   ├── Cd, Cl [, Cz]  (N,)
└── monitor/        per-diag_every time-series
    ├── step  (N,)
    ├── t     (N,)
    └── ...

Restart / append mode
---------------------
Pass ``append=True`` to ``open()`` to continue an existing run.
The file is opened in 'a' mode; grid and metadata are left intact;
existing dataset lengths are read so new rows are appended correctly.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    import h5py
except ImportError:
    raise ImportError("h5py is required.  Install with:  pip install h5py")


def _make_serializable(obj):
    """Recursively convert a config dict to a JSON-serializable structure."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return f"<ndarray:shape={obj.shape}>"   # store shape summary, not raw data
    if callable(obj):
        return f"<callable:{getattr(obj, '__name__', '?')}>"
    return obj


class H5Writer:
    """
    Append-friendly HDF5 writer for field snapshots and diagnostic time series.

    Usage::

        writer = H5Writer('output/run.h5')
        writer.open(config, grid)
        # ... simulation loop ...
        writer.write_snapshot(step, t, state)
        writer.append_diagnostics(step, t, Fx, Fy, Cd, Cl)
        writer.append_monitor(step, t, {'Nu': 8.4, 'ke': 0.12})
        writer.close()

    Restart (append) mode::

        writer = H5Writer('output/run.h5')
        writer.open(config, grid, append=True)  # opens existing file
        # simulation continues from where it left off
    """

    _DIAG_KEYS_2D = ('step', 't', 'Fx', 'Fy', 'Cd', 'Cl',
                      'Cd_p', 'Cd_v', 'Cl_p', 'Cl_v')
    _DIAG_KEYS_3D = ('step', 't', 'Fx', 'Fy', 'Fz', 'Cd', 'Cl', 'Cz',
                      'Cd_p', 'Cd_v', 'Cl_p', 'Cl_v')
    _CHUNK = 4096
    _VERSION = '0.5.0'
    _AXES  = ['x', 'y', 'z']
    _COMPS = ['u', 'v', 'w']

    def __init__(self, path: str):
        self.path = Path(path)
        self._file = None
        self._ndim = 2
        self._diag_keys = self._DIAG_KEYS_2D
        self._n_diag = 0
        self._n_monitor = 0

    # ── Context manager ────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Lifecycle ──────────────────────────────────────────────────────

    def open(self, config: dict, grid, append: bool = False) -> 'H5Writer':
        """
        Open (or re-open in append mode) the HDF5 file.

        Parameters
        ----------
        config : dict
            Simulation config (stored as JSON attribute on new files).
        grid : MACGrid
            Grid object (coordinates written on new files only).
        append : bool
            If True, open the existing file in append mode.  Existing datasets
            are preserved; new rows are appended after the existing data.
            If False (default), create a fresh file (overwrites any existing).
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if append:
            if not self.path.exists():
                raise FileNotFoundError(
                    f"Restart requested but file not found: {self.path}"
                )
            self._file = h5py.File(str(self.path), 'a')
            self._ndim = int(self._file['grid'].attrs.get('ndim', 2))
            self._diag_keys = self._DIAG_KEYS_3D if self._ndim == 3 else self._DIAG_KEYS_2D
            if 'diagnostics' in self._file:
                self._n_diag = int(self._file['diagnostics']['step'].shape[0])
            if 'monitor' in self._file:
                self._n_monitor = int(self._file['monitor']['step'].shape[0])
            return self

        # ── New file ──────────────────────────────────────────────────
        self._file = h5py.File(str(self.path), 'w')

        self._file.attrs['config_json'] = json.dumps(_make_serializable(config))
        self._file.attrs['created_at'] = datetime.now().isoformat()
        self._file.attrs['jax_cfd_version'] = self._VERSION

        self._ndim = grid.ndim
        self._diag_keys = self._DIAG_KEYS_3D if grid.ndim == 3 else self._DIAG_KEYS_2D

        grp = self._file.create_group('grid')
        grp.attrs['ndim'] = grid.ndim
        cc = grid.cell_centers()
        for ax in range(grid.ndim):
            grp.create_dataset(f'{self._AXES[ax]}c', data=np.array(cc[ax]))
            fc = grid.face_centers(ax)
            grp.create_dataset(f'{self._AXES[ax]}{self._COMPS[ax]}',
                               data=np.array(fc[ax]))

        self._file.create_group('fields')

        dgrp = self._file.create_group('diagnostics')
        for key in self._diag_keys:
            dtype = np.int64 if key == 'step' else np.float32
            dgrp.create_dataset(
                key, shape=(0,), maxshape=(None,), chunks=(self._CHUNK,), dtype=dtype,
            )

        return self

    def close(self):
        """Flush and close the HDF5 file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    # ── Write operations ───────────────────────────────────────────────

    def write_snapshot(self, step: int, t: float, state) -> None:
        """
        Save a full field snapshot (u, v, p[, T, ...]) under ``fields/step_XXXXXXX/``.
        Arrays are gzip-compressed (level 4) at float32 precision.
        """
        key = f'step_{step:07d}'
        grp = self._file['fields'].create_group(key)
        grp.attrs['step'] = int(step)
        grp.attrs['t'] = float(t)

        opts = dict(compression='gzip', compression_opts=4, dtype=np.float32)
        for i, name in enumerate(self._COMPS[:self._ndim]):
            grp.create_dataset(name, data=np.array(state.velocity[i]), **opts)
        grp.create_dataset('p', data=np.array(state.pressure), **opts)

        if state.scalars:
            for name, arr in state.scalars.items():
                grp.create_dataset(name, data=np.array(arr), **opts)

        self._file.flush()

    def append_diagnostics(self, step: int, t: float, **kwargs) -> None:
        """Append one row to the IBM force time-series.  Flushes every 512 calls.

        Accepts force/coefficient values as keyword arguments, e.g.:
            append_diagnostics(step, t, Fx=1.0, Fy=0.5, Cd=0.8, Cl=0.4)
            append_diagnostics(step, t, Fx=1.0, Fy=0.5, Fz=0.1, Cd=0.8, Cl=0.4, Cz=0.08)
        """
        dgrp = self._file['diagnostics']
        vals = dict(step=step, t=t, **kwargs)
        for key in self._diag_keys:
            ds = dgrp[key]
            n = ds.shape[0]
            ds.resize(n + 1, axis=0)
            ds[n] = vals.get(key, float('nan'))

        self._n_diag += 1
        if self._n_diag % 512 == 0:
            self._file.flush()

    def append_monitor(self, step: int, t: float, values: dict) -> None:
        """
        Append one row of monitor / probe data to the ``monitor/`` group.

        ``values`` is an arbitrary ``{str: float}`` dict, e.g.::

            {'Nu': 8.4, 'ke': 0.12, 'div': 1e-7,
             'probe_0_u': 0.35, 'probe_0_v': -0.02, 'probe_0_p': 0.01}

        Datasets are created dynamically on the first call.  Keys that appear
        after the group already exists are back-filled with NaN for previous
        rows so every column has the same length.
        Flushes every 512 calls.
        """
        if 'monitor' not in self._file:
            mgrp = self._file.create_group('monitor')
            for key in ['step', 't'] + list(values.keys()):
                dtype = np.int64 if key == 'step' else np.float32
                mgrp.create_dataset(
                    key, shape=(0,), maxshape=(None,), chunks=(self._CHUNK,), dtype=dtype,
                )

        mgrp = self._file['monitor']
        n = int(mgrp['step'].shape[0])

        mgrp['step'].resize(n + 1, axis=0)
        mgrp['step'][n] = int(step)
        mgrp['t'].resize(n + 1, axis=0)
        mgrp['t'][n] = float(t)

        for k, v in values.items():
            if k not in mgrp:
                # New key: create dataset back-filled with NaN up to current length
                ds = mgrp.create_dataset(
                    k, shape=(n + 1,), maxshape=(None,), chunks=(self._CHUNK,), dtype=np.float32,
                )
                ds[:n] = float('nan')
            else:
                mgrp[k].resize(n + 1, axis=0)
            mgrp[k][n] = float(v)

        self._n_monitor += 1
        if self._n_monitor % 512 == 0:
            self._file.flush()
