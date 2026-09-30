"""
HDF5 Reader for JAX-CFD simulation outputs.

Provides simple functions to load field snapshots and diagnostic time series
from files created by H5Writer.
"""

import json
import numpy as np
from typing import Dict, List, Optional

try:
    import h5py
except ImportError:
    raise ImportError("h5py is required.  Install with:  pip install h5py")


# ── Low-level helpers ──────────────────────────────────────────────────────

def _open(path: str):
    return h5py.File(str(path), 'r')


# ── Metadata ───────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Return the CONFIG dict that was stored when the run was created."""
    with _open(path) as f:
        return json.loads(f.attrs['config_json'])


def load_grid(path: str) -> Dict[str, np.ndarray]:
    """
    Return grid coordinate arrays.

    2D keys: ``xc``, ``yc``, ``xu``, ``yv``
    3D keys: also ``zc``, ``zw``
    """
    with _open(path) as f:
        return {k: f['grid'][k][:] for k in f['grid'].keys()}


def list_steps(path: str) -> List[int]:
    """Return sorted list of saved step indices (from the fields group)."""
    with _open(path) as f:
        keys = list(f['fields'].keys())
    steps = sorted(int(k.split('_')[1]) for k in keys)
    return steps


# ── Field snapshots ────────────────────────────────────────────────────────

def load_snapshot(path: str, step: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Load one field snapshot.

    Parameters
    ----------
    path:  Path to the HDF5 file.
    step:  Step index to load.  If None, loads the last saved step.

    Returns
    -------
    dict with keys: ``u``, ``v``, ``p``, ``t``, ``step``.
    """
    steps = list_steps(path)
    if not steps:
        raise ValueError(f"No snapshots found in {path}")
    if step is None:
        step = steps[-1]
    elif step not in steps:
        raise KeyError(f"Step {step} not found.  Available: {steps}")

    key = f'step_{step:07d}'
    with _open(path) as f:
        grp = f['fields'][key]
        snap = {
            'p'   : grp['p'][:],
            't'   : float(grp.attrs['t']),
            'step': int(grp.attrs['step']),
        }
        # Load velocity components dynamically (u, v, and w if 3D)
        for name in ('u', 'v', 'w'):
            if name in grp:
                snap[name] = grp[name][:]
        # Load any scalar fields present (e.g., 'T' for heat solver)
        _velocity = {'u', 'v', 'w', 'p'}
        for name in grp.keys():
            if name not in _velocity:
                snap[name] = grp[name][:]
        return snap


def load_all_snapshots(path: str) -> List[Dict[str, np.ndarray]]:
    """Load every saved snapshot in chronological order."""
    return [load_snapshot(path, s) for s in list_steps(path)]


# ── Diagnostic time series ─────────────────────────────────────────────────

def load_diagnostics(path: str) -> Dict[str, np.ndarray]:
    """
    Load the full diagnostic time series.

    Returns
    -------
    dict with keys: ``step``, ``t``, ``Fx``, ``Fy``, ``Cd``, ``Cl``.
    """
    with _open(path) as f:
        dgrp = f['diagnostics']
        return {k: dgrp[k][:] for k in dgrp.keys()}


def load_diagnostics_window(
    path: str,
    t_start: float = 0.0,
    t_end: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Load diagnostics within a time window [t_start, t_end].

    Useful for computing statistics after the initial transient.
    """
    diag = load_diagnostics(path)
    t = diag['t']
    mask = t >= t_start
    if t_end is not None:
        mask &= t <= t_end
    return {k: v[mask] for k, v in diag.items()}


# ── Monitor time series ────────────────────────────────────────────────────

def load_monitor(path: str) -> Dict[str, np.ndarray]:
    """
    Load the full monitor time series from the ``monitor/`` group.

    Returns a dict with at least ``step`` and ``t``, plus any quantities
    recorded by the runner (e.g. ``Nu``, ``ke``, ``div``, ``probe_0_u``, ...).
    Returns an empty dict if no monitor data exists in the file.
    """
    with _open(path) as f:
        if 'monitor' not in f:
            return {}
        mgrp = f['monitor']
        return {k: mgrp[k][:] for k in mgrp.keys()}


def load_monitor_window(
    path: str,
    t_start: float = 0.0,
    t_end: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Load monitor data within a time window [t_start, t_end].

    Useful for computing time-averaged Nu or ke after the initial transient.

    Example
    -------
    mon = load_monitor_window('output/run.h5', t_start=50.0)
    print(f"Mean Nu = {mon['Nu'].mean():.3f}")
    """
    mon = load_monitor(path)
    if not mon:
        return {}
    t = mon['t']
    mask = t >= t_start
    if t_end is not None:
        mask &= t <= t_end
    return {k: v[mask] for k, v in mon.items()}
