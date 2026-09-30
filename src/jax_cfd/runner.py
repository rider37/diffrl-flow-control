"""
Universal simulation runner for JAX-CFD.

Usage (Python API)
------------------
    from jax_cfd.runner import run
    from examples.rayleigh_taylor import CONFIG
    run(CONFIG)

    # Restart from a previous checkpoint:
    run(CONFIG, restart_from='examples/rayleigh_taylor/output/run.h5')

Usage (CLI)
-----------
    diffflow examples/rayleigh_taylor
    diffflow examples/rayleigh_taylor --restart

Supported solvers
-----------------
    'incompressible_ns'             -- ∇·u=0, Du/Dt = -∇p/ρ + ν∇²u
    'incompressible_ns_heat'        -- above + DT/Dt = α∇²T  (Boussinesq buoyancy)
    'incompressible_ns_phase_field'      -- above + Cahn-Hilliard φ  (surface tension + density buoyancy)
    'incompressible_ns_heat_phase_field' -- above + CH φ + T  (boiling: phase change + dual buoyancy)
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from functools import partial

# ── Persistent JIT compilation cache ────────────────────────────────────
# Saves compiled XLA programs to disk. Same grid + config → instant startup.
# Override with JAX_COMPILATION_CACHE_DIR env variable.
import os as _os
_cache_dir = _os.environ.get(
    'JAX_COMPILATION_CACHE_DIR',
    _os.path.join(_os.path.expanduser('~'), '.jax_cache'),
)
jax.config.update("jax_compilation_cache_dir", _cache_dir)

from jax_cfd.core.advection import advect_scalar_muscl
from jax_cfd.core.boundaries import apply_boundaries, pad_velocity_for_diffusion
from jax_cfd.core.geometry import get_geometry, compute_normals_2d
from jax_cfd.core.ibm import (apply_ibm_forcing, compute_tensor_weights_mac,
                               apply_ibm_wall_model, compute_probe_points)
from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.operators import gradient_p as _gradient_p_base
from jax_cfd.core.probes import extract_ibm_forces, extract_probe_data
from jax_cfd.core.schemes import (
    get_pressure_solver, make_advection_scheme,
    make_pressure_solver_hybrid, make_pressure_solver_fft_periodic,
)
from jax_cfd.core.solver import create_step_fn
from jax_cfd.core.state import FluidState
from jax_cfd.core.turbulence import get_turbulence_model
from jax_cfd.io.writer import H5Writer


# ── Monitor quantity computers ──────────────────────────────────────────────

def _compute_monitor(
    quantities: list,
    state: FluidState,
    grid: MACGrid,
    q_ref: float,
    heat_cfg: dict = None,
) -> dict:
    """
    Compute monitor quantities from the current FluidState.

    All computations stay on-device as JAX arrays.  A single
    ``jax.device_get`` at the end transfers everything to the host in one
    batch, avoiding per-quantity device→host synchronisation overhead.

    Supported keys
    --------------
    'Cd'  drag coefficient            (requires IBM)
    'Cl'  lift coefficient            (requires IBM)
    'ke'  mean kinetic energy         0.5 * mean(u² + v²)
    'div' RMS velocity divergence     continuity check, should be ~1e-6
    'Nu'  Nusselt number              wall temperature-gradient method
    """
    from jax_cfd.core.operators import divergence as _div_op
    vals_jax: dict = {}
    for qty in quantities:
        if qty == 'Cd':
            vals_jax['Cd'] = (jnp.asarray(state.ibm_forces)[0] / q_ref
                              if state.ibm_forces is not None else jnp.nan)
        elif qty == 'Cl':
            vals_jax['Cl'] = (jnp.asarray(state.ibm_forces)[1] / q_ref
                              if state.ibm_forces is not None else jnp.nan)
        elif qty == 'ke':
            vals_jax['ke'] = 0.5 * sum(jnp.mean(u ** 2) for u in state.velocity)
        elif qty == 'div':
            d = _div_op(state.velocity, grid)
            vals_jax['div'] = jnp.sqrt(jnp.mean(d ** 2))
        elif qty == 'Nu':
            T = state.scalars.get('T') if state.scalars else None
            if T is not None and heat_cfg is not None:
                T_hot  = float(heat_cfg.get('T_hot',  1.0))
                T_cold = float(heat_cfg.get('T_cold', 0.0))
                dT     = T_hot - T_cold
                wall_axis = int(heat_cfg.get('wall_axis', grid.ndim - 1))
                dy_arr = grid.spacings_array(wall_axis)
                if isinstance(dy_arr, (int, float)):
                    dy_bot = float(dy_arr)
                    dy_top = float(dy_arr)
                else:
                    dy_bot = float(dy_arr[0])
                    dy_top = float(dy_arr[-1])
                sl_bot = [slice(None)] * grid.ndim
                sl_top = [slice(None)] * grid.ndim
                sl_bot[wall_axis] = 0
                sl_top[wall_axis] = -1
                Lw = float(grid.domain_size[wall_axis])
                Nu_bot = jnp.mean(2.0 * Lw * (T_hot  - T[tuple(sl_bot)]) / (dy_bot * dT))
                Nu_top = jnp.mean(2.0 * Lw * (T[tuple(sl_top)] - T_cold) / (dy_top * dT))
                vals_jax['Nu'] = 0.5 * (Nu_bot + Nu_top)
            else:
                vals_jax['Nu'] = jnp.nan
    # Single device→host transfer for all quantities
    return {k: float(v) for k, v in jax.device_get(vals_jax).items()}


# ── Solver registry ────────────────────────────────────────────────────────

_REQUIRED_KEYS: dict[str, list[str]] = {
    'incompressible_ns':             ['Re'],
    'incompressible_ns_heat':        ['Re', 'Pr'],
    'incompressible_ns_phase_field': ['Re'],
    'incompressible_ns_heat_phase_field': ['Re', 'Pr'],
    'compressible_euler':            [],
    'compressible_ns':               [],
}


# ── Validation ─────────────────────────────────────────────────────────────

_VALID_TURBULENCE = {'laminar', 'smagorinsky', 'wale', 'dynamic_smagorinsky', 'k_omega_sst'}
_VALID_TIME_SCHEMES = {'rk3', 'euler'}
_VALID_ADVECTION = {'muscl', 'central', 'weno5'}
_VALID_DIFFUSION = {'explicit', 'implicit'}
_VALID_BC_TYPES = {
    'wall_no_slip', 'wall_slip', 'wall_free_slip', 'wall_function',
    'fixedValue', 'inlet', 'inlet_parabolic', 'outlet', 'outlet_zero_gradient', 'periodic',
    'symmetry', 'zeroGradient',
}


def _validate(cfg: dict) -> None:
    """Validate config dict before simulation.

    Checks solver type, required keys, domain geometry, physical
    parameters, time control, boundary conditions, and scheme settings.
    """
    # ── Solver type ───────────────────────────────────────────────────
    solver = cfg.get('solver', 'incompressible_ns')
    if solver not in _REQUIRED_KEYS:
        raise ValueError(
            f"Unknown solver '{solver}'.  Valid options: {list(_REQUIRED_KEYS)}"
        )
    missing = [k for k in _REQUIRED_KEYS[solver] if k not in cfg]
    if missing:
        raise ValueError(
            f"solver='{solver}' requires config keys: {missing}"
        )

    # ── Domain geometry ───────────────────────────────────────────────
    dom = cfg.get('domain')
    if dom is None:
        raise ValueError("Config must contain a 'domain' dict.")
    cells = dom.get('cells')
    size = dom.get('size')
    if cells is None:
        raise ValueError("domain.cells is required (e.g. [64, 64]).")
    if size is None:
        raise ValueError("domain.size is required (e.g. [1.0, 1.0]).")
    if len(cells) != len(size):
        raise ValueError(
            f"domain.cells length ({len(cells)}) != domain.size length ({len(size)})."
        )
    ndim = len(cells)
    if ndim not in (2, 3):
        raise ValueError(f"Only 2D and 3D supported, got ndim={ndim}.")
    for i, n in enumerate(cells):
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"domain.cells[{i}]={n} must be a positive integer.")
    for i, s in enumerate(size):
        if float(s) <= 0:
            raise ValueError(f"domain.size[{i}]={s} must be positive.")

    # ── Physical parameters ───────────────────────────────────────────
    if solver.startswith('compressible'):
        if 'gamma' in cfg and float(cfg['gamma']) <= 0:
            raise ValueError(f"gamma={cfg['gamma']} must be positive.")
        return  # compressible solver has its own validation
    if 'Re' in cfg and float(cfg['Re']) <= 0:
        raise ValueError(f"Re={cfg['Re']} must be positive.")
    if 'Pr' in cfg and float(cfg['Pr']) <= 0:
        raise ValueError(f"Pr={cfg['Pr']} must be positive.")
    # ── Solver-specific sub-config checks ────────────────────────────
    if solver in ('incompressible_ns_phase_field', 'incompressible_ns_heat_phase_field'):
        if 'phase_field' not in cfg:
            raise ValueError(
                f"solver='{solver}' requires a 'phase_field' config dict."
            )
    if solver == 'incompressible_ns_heat_phase_field':
        if 'heat' not in cfg:
            raise ValueError(
                "solver='incompressible_ns_heat_phase_field' requires a 'heat' config dict."
            )

    if 'Ra' in cfg and float(cfg['Ra']) <= 0:
        raise ValueError(f"Ra={cfg['Ra']} must be positive.")

    # ── Time control ──────────────────────────────────────────────────
    tc = cfg.get('time_control', {})
    if 'max_cfl' in tc and float(tc['max_cfl']) <= 0:
        raise ValueError(f"time_control.max_cfl={tc['max_cfl']} must be positive.")
    if 't_end' in tc and float(tc['t_end']) <= 0:
        raise ValueError(f"time_control.t_end={tc['t_end']} must be positive.")
    if 'diag_every' in tc and int(tc['diag_every']) <= 0:
        raise ValueError(f"time_control.diag_every must be positive.")
    if 'save_every' in tc and int(tc['save_every']) <= 0:
        raise ValueError(f"time_control.save_every must be positive.")

    # ── Boundary conditions ───────────────────────────────────────────
    bc_cfg = cfg.get('boundaries', {})
    axis_names = ['x', 'y', 'z'][:ndim]
    for ax_name in axis_names:
        lo = bc_cfg.get(f'{ax_name}_min', {})
        hi = bc_cfg.get(f'{ax_name}_max', {})
        lo_type = lo.get('type', '') if isinstance(lo, dict) else ''
        hi_type = hi.get('type', '') if isinstance(hi, dict) else ''
        # Check known BC types
        for label, btype in [(f'{ax_name}_min', lo_type), (f'{ax_name}_max', hi_type)]:
            if btype and btype not in _VALID_BC_TYPES:
                raise ValueError(
                    f"Unknown BC type '{btype}' on {label}.  "
                    f"Valid: {sorted(_VALID_BC_TYPES)}"
                )
        # Periodic must be paired
        if (lo_type == 'periodic') != (hi_type == 'periodic'):
            raise ValueError(
                f"Periodic BC on {ax_name} axis must be set on both "
                f"{ax_name}_min and {ax_name}_max."
            )

    # ── Schemes ───────────────────────────────────────────────────────
    sch = cfg.get('schemes', {})
    ts = sch.get('time', 'rk3').lower()
    if ts not in _VALID_TIME_SCHEMES:
        raise ValueError(f"Unknown time scheme '{ts}'.  Valid: {sorted(_VALID_TIME_SCHEMES)}")
    adv = sch.get('advection', 'muscl').lower()
    if adv not in _VALID_ADVECTION:
        raise ValueError(f"Unknown advection scheme '{adv}'.  Valid: {sorted(_VALID_ADVECTION)}")
    diff = sch.get('diffusion', 'explicit').lower()
    if diff not in _VALID_DIFFUSION:
        raise ValueError(f"Unknown diffusion scheme '{diff}'.  Valid: {sorted(_VALID_DIFFUSION)}")

    # ── Turbulence model ──────────────────────────────────────────────
    turb_model = cfg.get('turbulence', {}).get('model', 'laminar').lower()
    if turb_model not in _VALID_TURBULENCE:
        raise ValueError(
            f"Unknown turbulence model '{turb_model}'.  "
            f"Valid: {sorted(_VALID_TURBULENCE)}"
        )


# ── Component builders ─────────────────────────────────────────────────────

def _build_grid(cfg: dict) -> MACGrid:
    dom = cfg['domain']
    stretch = dom.get('stretch', None)
    return MACGrid(tuple(dom['cells']), tuple(dom['size']), stretch=stretch)


def _build_physics_params(cfg: dict):
    """Return (nu, U_ref, L_ref, rho)."""
    ref   = cfg.get('reference', {})
    U_ref = float(ref.get('U',   1.0))
    L_ref = float(ref.get('L',   1.0))
    rho   = float(ref.get('rho', 1.0))
    Re    = float(cfg['Re'])
    nu    = U_ref * L_ref / Re
    return nu, U_ref, L_ref, rho


def _get_periodic_axes(bc_cfg: dict, ndim: int) -> tuple:
    """Return the axes (0-indexed) whose BC type is 'periodic'.

    Segmented BCs (list) are never periodic, so they are skipped.
    """
    axes_names = ['x', 'y', 'z']
    def _is_periodic(face_key):
        bc = bc_cfg.get(face_key, {})
        if isinstance(bc, dict):
            return bc.get('type') == 'periodic'
        return False  # segmented BC list → not periodic
    return tuple(
        i for i in range(ndim)
        if _is_periodic(f'{axes_names[i]}_min')
        or _is_periodic(f'{axes_names[i]}_max')
    )


def _build_closures(cfg: dict, grid: MACGrid):
    """Return (turb_fn, advect_fn, solve_p, apply_bc, pad_diff, gradient_p_fn)."""
    bc_cfg   = cfg.get('boundaries', {})
    sch_cfg  = cfg.get('schemes', {})
    turb_cfg = cfg.get('turbulence', {'model': 'laminar'})

    # Inject ν into wall_function BC entries so the ghost-cell computation
    # has access to the viscosity without changing the solver call signature.
    nu = 1.0 / float(cfg.get('Re', 100.0))
    for key, bc in bc_cfg.items():
        if isinstance(bc, dict) and bc.get('type') == 'wall_function':
            bc['nu'] = nu

    ndim     = grid.ndim
    periodic = _get_periodic_axes(bc_cfg, ndim)

    # Pass periodic_axes + user-specified params (e.g. Cs, Cw, Cs_max) to the model
    turb_model_name = turb_cfg.get('model', 'laminar')
    _turb_params = {k: v for k, v in turb_cfg.items() if k != 'model'}
    _turb_params['periodic_axes'] = periodic

    # k-ω SST returns (scalar_rhs_fn, get_nu_t_fn, init_scalars) — handled specially
    _rans_rhs_fn = None
    _rans_init_scalars = None
    if turb_model_name == 'k_omega_sst':
        from jax_cfd.core.rans import make_k_omega_sst
        _rans_rhs_fn, turb_fn, _rans_init_scalars = make_k_omega_sst(grid, nu)
    else:
        turb_fn = get_turbulence_model(turb_model_name, **_turb_params)

    advect_fn = make_advection_scheme(sch_cfg.get('advection', 'muscl'), periodic)

    # Pressure solver: auto-select based on grid type and periodic axes
    if not grid.is_uniform:
        solve_p = make_pressure_solver_hybrid(grid, periodic_axes=periodic)
    elif periodic:
        solve_p = make_pressure_solver_fft_periodic(periodic)
    else:
        solve_p = get_pressure_solver(sch_cfg.get('pressure', 'fft'))

    gradient_p_fn = partial(_gradient_p_base, periodic_axes=periodic) if periodic else None

    def apply_bc(vel, _cfg, g, inlet_vel=None):
        return apply_boundaries(vel, bc_cfg, g, inlet_vel=inlet_vel)

    def pad_diff(vel, g, inlet_vel=None):
        return pad_velocity_for_diffusion(vel, bc_cfg, g, inlet_vel=inlet_vel)

    return turb_fn, advect_fn, solve_p, apply_bc, pad_diff, gradient_p_fn, _rans_rhs_fn, _rans_init_scalars


def _build_ibm(cfg: dict, grid: MACGrid):
    """
    Build IBM closure(s) for all bodies in ``cfg['bodies']``.

    Supports multiple bodies: corrections are applied sequentially and forces
    are summed to produce a single (ndim,) force vector (backward compatible
    with single-body diagnostics).

    **Scan optimization**: When *all* bodies are scannable (static or rotating,
    i.e. cached weights and constant target velocity), they are batched into a
    single ``jax.lax.scan`` loop.  This traces the IBM subgraph only once
    regardless of body count, yielding significantly faster JIT compilation
    for large body counts (2-3x at 25 bodies, more at higher counts).

    When *any* dynamic body is present (oscillating, translating, pitching,
    orbital), the classic per-body for-loop is used for all bodies so that
    config ordering is preserved exactly.

    Returns ``(ibm_fn, coords_list, ds_list)`` or ``(None, [], [])`` if no bodies.
    """
    bodies = cfg.get('bodies', [])
    if not bodies:
        return None, [], []

    ndim = grid.ndim
    coords_list = []
    ds_list     = []

    # ── Classify bodies ───────────────────────────────────────────────────
    has_dynamic = any(
        b.get('motion', {}).get('type', 'static')
        not in ('static', 'rotating')
        for b in bodies
    )
    has_wall_model = any(
        b.get('wall_model', {}).get('enabled', False)
        for b in bodies
    )

    # ── Helper: build a per-body closure (for-loop path) ──────────────────
    def _make_body_fn(coords_, ds_, motion_cfg, cached_w, ndim_):
        mtype = motion_cfg.get('type', 'static')
        n_lag = coords_[0].shape[0]

        if mtype == 'static':
            U_tgt = tuple(jnp.zeros(n_lag) for _ in range(ndim_))
            def fn(vel, _t, g, dt_sub):
                vel_c, dU = apply_ibm_forcing(vel, U_tgt, coords_, g,
                                              weights=cached_w)
                return vel_c, extract_ibm_forces(dU, ds_, g, dt_sub)
            return fn

        if mtype == 'rotating':
            omega = float(motion_cfg.get('omega', 0.0))
            center_ = tuple(float(jnp.mean(c)) for c in coords_)
            if ndim_ == 2:
                U_tgt = (
                    -omega * (coords_[1] - center_[1]),
                     omega * (coords_[0] - center_[0]),
                )
            else:
                ax_dir = motion_cfg.get('rotation_axis', [0.0, 0.0, 1.0])
                ax = jnp.array(ax_dir[:3], dtype=jnp.float32) * omega
                r = tuple(coords_[i] - center_[i] for i in range(3))
                U_tgt = (
                    ax[1] * r[2] - ax[2] * r[1],
                    ax[2] * r[0] - ax[0] * r[2],
                    ax[0] * r[1] - ax[1] * r[0],
                )
            def fn(vel, _t, g, dt_sub):
                vel_c, dU = apply_ibm_forcing(vel, U_tgt, coords_, g,
                                              weights=cached_w)
                return vel_c, extract_ibm_forces(dU, ds_, g, dt_sub)
            return fn

        if mtype == 'oscillating':
            amp  = jnp.array(motion_cfg.get('amplitude',  [0.0] * ndim_))
            freq = jnp.array(motion_cfg.get('frequency',  [0.0] * ndim_))
            coords0_ = coords_
            def fn(vel, t, g, dt_sub):
                phase = 2.0 * jnp.pi * freq * t
                coords_t = tuple(
                    c0 + amp[i] * jnp.sin(phase[i])
                    for i, c0 in enumerate(coords0_)
                )
                U_tgt = tuple(
                    amp[i] * 2.0 * jnp.pi * freq[i] * jnp.cos(phase[i])
                    * jnp.ones(n_lag)
                    for i in range(ndim_)
                )
                vel_c, dU = apply_ibm_forcing(vel, U_tgt, coords_t, g)
                return vel_c, extract_ibm_forces(dU, ds_, g, dt_sub)
            return fn

        if mtype == 'translating':
            vb = jnp.array(motion_cfg.get('velocity', [0.0] * ndim_))
            coords0_ = coords_
            U_tgt = tuple(vb[i] * jnp.ones(n_lag) for i in range(ndim_))
            def fn(vel, t, g, dt_sub):
                coords_t = tuple(
                    c0 + vb[i] * t for i, c0 in enumerate(coords0_)
                )
                vel_c, dU = apply_ibm_forcing(vel, U_tgt, coords_t, g)
                return vel_c, extract_ibm_forces(dU, ds_, g, dt_sub)
            return fn

        if mtype == 'pitching':
            if ndim_ != 2:
                raise NotImplementedError("Pitching motion is 2D only.")
            amp_rad = float(np.radians(float(motion_cfg.get('amplitude_deg', 10.0))))
            freq = float(motion_cfg.get('frequency', 1.0))
            pivot_ = motion_cfg.get('pivot', None)
            if pivot_ is None:
                pivot_ = tuple(float(jnp.mean(c)) for c in coords_)
            else:
                pivot_ = tuple(float(p) for p in pivot_)
            r0_x = coords_[0] - pivot_[0]
            r0_y = coords_[1] - pivot_[1]
            def fn(vel, t, g, dt_sub):
                theta = amp_rad * jnp.sin(2.0 * jnp.pi * freq * t)
                omega = amp_rad * 2.0 * jnp.pi * freq * jnp.cos(2.0 * jnp.pi * freq * t)
                cos_t = jnp.cos(theta)
                sin_t = jnp.sin(theta)
                dx_ = r0_x * cos_t - r0_y * sin_t
                dy_ = r0_x * sin_t + r0_y * cos_t
                coords_t = (pivot_[0] + dx_, pivot_[1] + dy_)
                U_tgt = (-omega * dy_, omega * dx_)
                vel_c, dU = apply_ibm_forcing(vel, U_tgt, coords_t, g)
                return vel_c, extract_ibm_forces(dU, ds_, g, dt_sub)
            return fn

        if mtype == 'orbital':
            omega = float(motion_cfg.get('omega', 1.0))
            oc = motion_cfg.get('orbit_center', None)
            if oc is None:
                raise ValueError("'orbital' motion requires 'orbit_center'.")
            oc = tuple(float(c) for c in oc)
            r0 = tuple(coords_[i] - oc[i] for i in range(ndim_))

            if ndim_ == 2:
                def fn(vel, t, g, dt_sub):
                    phi = omega * t
                    cos_p = jnp.cos(phi)
                    sin_p = jnp.sin(phi)
                    dx_ = r0[0] * cos_p - r0[1] * sin_p
                    dy_ = r0[0] * sin_p + r0[1] * cos_p
                    coords_t = (oc[0] + dx_, oc[1] + dy_)
                    U_tgt = (-omega * dy_, omega * dx_)
                    vel_c, dU = apply_ibm_forcing(vel, U_tgt, coords_t, g)
                    return vel_c, extract_ibm_forces(dU, ds_, g, dt_sub)
            else:
                def fn(vel, t, g, dt_sub):
                    phi = omega * t
                    cos_p = jnp.cos(phi)
                    sin_p = jnp.sin(phi)
                    dx_ = r0[0] * cos_p - r0[1] * sin_p
                    dy_ = r0[0] * sin_p + r0[1] * cos_p
                    coords_t = (oc[0] + dx_, oc[1] + dy_, oc[2] + r0[2])
                    U_tgt = (-omega * dy_, omega * dx_,
                             jnp.zeros(n_lag))
                    vel_c, dU = apply_ibm_forcing(vel, U_tgt, coords_t, g)
                    return vel_c, extract_ibm_forces(dU, ds_, g, dt_sub)
            return fn

        raise ValueError(f"Unknown motion type: {mtype}")

    # ── Path A: all scannable → jax.lax.scan ──────────────────────────────
    if not has_dynamic and not has_wall_model:
        scannable_data = []   # (coords, ds, weights, U_tgt)

        for body_cfg in bodies:
            coords, ds_arr = get_geometry(body_cfg, grid)
            if len(coords) != ndim:
                raise ValueError(
                    f"Geometry returned {len(coords)} coordinate arrays "
                    f"but grid is {ndim}D."
                )
            coords_list.append(coords)
            ds_list.append(ds_arr)

            motion_cfg = body_cfg.get('motion', {'type': 'static'})
            mtype = motion_cfg.get('type', 'static')
            cached_w = compute_tensor_weights_mac(grid, coords)

            if mtype == 'static':
                n_lag = coords[0].shape[0]
                U_tgt = tuple(jnp.zeros(n_lag) for _ in range(ndim))
            else:  # rotating
                omega = float(motion_cfg.get('omega', 0.0))
                center_ = tuple(float(jnp.mean(c)) for c in coords)
                if ndim == 2:
                    U_tgt = (
                        -omega * (coords[1] - center_[1]),
                         omega * (coords[0] - center_[0]),
                    )
                else:
                    ax_dir = motion_cfg.get('rotation_axis', [0.0, 0.0, 1.0])
                    ax = jnp.array(ax_dir[:3], dtype=jnp.float32) * omega
                    r = tuple(coords[i] - center_[i] for i in range(3))
                    U_tgt = (
                        ax[1] * r[2] - ax[2] * r[1],
                        ax[2] * r[0] - ax[0] * r[2],
                        ax[0] * r[1] - ax[1] * r[0],
                    )
            scannable_data.append((coords, ds_arr, cached_w, U_tgt))

        n_scan = len(scannable_data)
        max_n_pts = max(d[0][0].shape[0] for d in scannable_data)

        def _pad_2d(arr, target_rows):
            return jnp.pad(arr, ((0, target_rows - arr.shape[0]), (0, 0)))

        def _pad_1d(arr, target_len):
            return jnp.pad(arr, (0, target_len - arr.shape[0]))

        # Stack weights: stacked_w[cmp][ax] = (n_scan, max_n_pts, N_ax)
        stacked_w = tuple(
            tuple(
                jnp.stack([_pad_2d(d[2][cmp][ax], max_n_pts)
                           for d in scannable_data], axis=0)
                for ax in range(ndim)
            )
            for cmp in range(ndim)
        )
        stacked_U_tgt = tuple(
            jnp.stack([_pad_1d(d[3][cmp], max_n_pts)
                       for d in scannable_data], axis=0)
            for cmp in range(ndim)
        )
        stacked_ds = jnp.stack(
            [_pad_1d(d[1], max_n_pts) for d in scannable_data], axis=0
        )
        stacked_masks = jnp.stack(
            [jnp.arange(max_n_pts) < d[0][0].shape[0]
             for d in scannable_data], axis=0
        )
        dv = jnp.prod(jnp.array(grid.spacings))

        if ndim == 2:
            def _scan_body(carry_vel, body_idx):
                mask = stacked_masks[body_idx]
                # Fetch weights once per component
                W = [(stacked_w[c][0][body_idx], stacked_w[c][1][body_idx])
                     for c in range(2)]
                # Interpolate
                u_lag = [jnp.einsum('ki,ij,kj->k', W[c][0], carry_vel[c], W[c][1]) * dv
                         for c in range(2)]
                # dU = U_tgt - U_interp, masked for padding
                dU = [jnp.where(mask, stacked_U_tgt[c][body_idx] - u_lag[c], 0.0)
                      for c in range(2)]
                # Spread
                du_eul = [jnp.einsum('ki,k,kj->ij', W[c][0], dU[c] * dv, W[c][1])
                          for c in range(2)]
                new_vel = tuple(carry_vel[c] + du_eul[c] for c in range(2))
                ds_b = stacked_ds[body_idx]
                forces = jnp.array([-jnp.sum(dU[c] * ds_b) for c in range(2)])
                return new_vel, forces
        else:  # 3D
            def _scan_body(carry_vel, body_idx):
                mask = stacked_masks[body_idx]
                W = [(stacked_w[c][0][body_idx],
                      stacked_w[c][1][body_idx],
                      stacked_w[c][2][body_idx]) for c in range(3)]
                u_lag = [jnp.einsum('ki,ijm,kj,km->k',
                                    W[c][0], carry_vel[c], W[c][1], W[c][2]) * dv
                         for c in range(3)]
                dU = [jnp.where(mask, stacked_U_tgt[c][body_idx] - u_lag[c], 0.0)
                      for c in range(3)]
                du_eul = [jnp.einsum('ki,k,kj,km->ijm',
                                     W[c][0], dU[c] * dv, W[c][1], W[c][2])
                          for c in range(3)]
                new_vel = tuple(carry_vel[c] + du_eul[c] for c in range(3))
                ds_b = stacked_ds[body_idx]
                forces = jnp.array([-jnp.sum(dU[c] * ds_b) for c in range(3)])
                return new_vel, forces

        def ibm_fn(vel, _t, _g, dt_sub):
            final_vel, per_body_forces = jax.lax.scan(
                _scan_body, vel, jnp.arange(n_scan)
            )
            total_forces = jnp.sum(per_body_forces, axis=0) / dt_sub
            return final_vel, total_forces

        return ibm_fn, coords_list, ds_list

    # ── Path B: mixed or all-dynamic → per-body for-loop ─────────────────
    body_fns = []

    # Compute molecular nu from config (needed for wall-model bodies)
    if has_wall_model:
        _ref = cfg.get('reference', {})
        _U = float(_ref.get('U', 1.0))
        _L = float(_ref.get('L', 1.0))
        _nu = _U * _L / float(cfg['Re'])

    for body_cfg in bodies:
        coords, ds_arr = get_geometry(body_cfg, grid)
        if len(coords) != ndim:
            raise ValueError(
                f"Geometry returned {len(coords)} coordinate arrays "
                f"but grid is {ndim}D."
            )
        coords_list.append(coords)
        ds_list.append(ds_arr)

        wm_cfg = body_cfg.get('wall_model', {})
        if wm_cfg.get('enabled', False):
            # ── Wall-modeled IBM closure (2D only) ────────────────────
            if ndim != 2:
                raise NotImplementedError(
                    "Wall-modeled IBM is currently 2D only. "
                    f"Got {ndim}D grid.")
            motion_type = body_cfg.get('motion', {}).get('type', 'static')
            if motion_type not in ('static',):
                raise ValueError(
                    f"Wall-modeled IBM requires static bodies, "
                    f"got motion type '{motion_type}'.")
            normals, _ = compute_normals_2d(coords)
            probe_factor = float(wm_cfg.get('probe_factor', 2.0))
            penalty = float(wm_cfg.get('penalty_factor', 0.1))
            probe_dist = probe_factor * min(grid.spacings)
            probe_coords = compute_probe_points(coords, normals, probe_dist)
            cached_w = compute_tensor_weights_mac(grid, coords)
            probe_w = compute_tensor_weights_mac(grid, probe_coords)

            def _make_wm_fn(coords_=coords, normals_=normals, ds_=ds_arr,
                            w_=cached_w, pw_=probe_w, nu_=_nu,
                            pd_=probe_dist, pf_=penalty):
                def fn(vel, _t, g, dt_sub):
                    vel_c, dU = apply_ibm_wall_model(
                        vel, coords_, normals_, ds_, g, nu_,
                        dt=dt_sub, probe_distance=pd_,
                        penalty_factor=pf_,
                        weights=w_, probe_weights=pw_)
                    return vel_c, extract_ibm_forces(dU, ds_, g, dt_sub)
                return fn

            body_fns.append(_make_wm_fn())
        else:
            # ── Standard direct-forcing closure ───────────────────────
            motion_cfg = body_cfg.get('motion', {'type': 'static'})
            mtype = motion_cfg.get('type', 'static')
            cached_w = (compute_tensor_weights_mac(grid, coords)
                        if mtype in ('static', 'rotating') else None)
            body_fns.append(_make_body_fn(coords, ds_arr, motion_cfg, cached_w, ndim))

    def ibm_fn(vel, t, g, dt_sub):
        total = jnp.zeros(ndim)
        for fn in body_fns:
            vel, f = fn(vel, t, g, dt_sub)
            total = total + f
        return vel, total

    return ibm_fn, coords_list, ds_list


def _build_ibm_scalar(cfg: dict, grid: MACGrid):
    """Build IBM scalar forcing for bodies with 'T' (Dirichlet) or 'q' (Neumann).

    Returns
    -------
    ibm_scalar_fn : callable or None
        Direct forcing closure for Dirichlet bodies ('T' key).
    scalar_source : dict or None
        Pre-computed volumetric heat source {'T': field} for Neumann bodies ('q' key).
    """
    from jax_cfd.core.ibm import (
        compute_weights_cell_center, apply_ibm_scalar_forcing,
        compute_ibm_heat_source_field,
    )
    bodies = cfg.get('bodies', [])

    # --- Dirichlet bodies ('T' key) → ibm_scalar_fn ---
    # Skip bodies with 'thermal_mass' — they use the penalty approach instead.
    dirichlet = [(b, get_geometry(b, grid)) for b in bodies
                 if 'T' in b and 'thermal_mass' not in b]

    ibm_scalar_fn = None
    if dirichlet:
        def _make_fn(coords_, T_wall_, w_cache_):
            def fn(scalars):
                T = scalars.get('T')
                if T is None:
                    return scalars
                T_new = apply_ibm_scalar_forcing(T, T_wall_, coords_, grid, weights=w_cache_)
                return {**scalars, 'T': T_new}
            return fn

        scalar_fns = []
        for body_cfg, (coords, _ds) in dirichlet:
            T_wall = float(body_cfg['T'])
            mtype = body_cfg.get('motion', {}).get('type', 'static')
            w_cache = compute_weights_cell_center(grid, coords) if mtype in ('static', 'rotating') else None
            scalar_fns.append(_make_fn(coords, T_wall, w_cache))

        def ibm_scalar_fn(scalars):
            for fn in scalar_fns:
                scalars = fn(scalars)
            return scalars

    # --- Neumann bodies ('q' key) → scalar_source ---
    neumann = [(b, get_geometry(b, grid)) for b in bodies if 'q' in b]

    scalar_source = None
    if neumann:
        Q_total = jnp.zeros(grid.cell_shape())
        for body_cfg, (coords, ds) in neumann:
            q_flux = float(body_cfg['q'])
            mtype = body_cfg.get('motion', {}).get('type', 'static')
            w_cache = compute_weights_cell_center(grid, coords) if mtype in ('static', 'rotating') else None
            Q_total = Q_total + compute_ibm_heat_source_field(
                q_flux, coords, grid, ds, weights=w_cache,
            )
        scalar_source = {'T': Q_total}

    return ibm_scalar_fn, scalar_source


def _build_inlet_schedule(cfg: dict, grid: MACGrid):
    """Build a JAX-compatible time-varying inlet velocity schedule.

    Config example::

        'inlet_schedule': {
            'type': 'ramp',
            'velocity_off': [0.0, 0.01],    # stagnant (tiny residual)
            'velocity_on':  [0.0, 1.5],     # peak inlet velocity
            't_on':  1.0,                   # centre of ramp-up
            't_off': 20.0,                  # centre of ramp-down
            'ramp_width': 0.5,              # tanh transition half-width
        }

    Returns None if no 'inlet_schedule' key in config.
    """
    schedule_cfg = cfg.get('inlet_schedule', None)
    if schedule_cfg is None:
        return None

    stype = schedule_cfg.get('type', 'ramp')
    ndim = grid.ndim

    if stype == 'ramp':
        v_off = jnp.array(
            schedule_cfg.get('velocity_off', [0.0] * ndim), dtype=jnp.float32)
        v_on = jnp.array(
            schedule_cfg.get('velocity_on', [0.0] * ndim), dtype=jnp.float32)
        t_on = float(schedule_cfg['t_on'])
        t_off = float(schedule_cfg['t_off'])
        w = float(schedule_cfg.get('ramp_width', 0.5))

        def schedule_fn(t):
            ramp_up = 0.5 * (1.0 + jnp.tanh((t - t_on) / w))
            ramp_down = 0.5 * (1.0 + jnp.tanh((t_off - t) / w))
            alpha = ramp_up * ramp_down
            vel = v_off + (v_on - v_off) * alpha
            return tuple(vel[i] for i in range(ndim))

        return schedule_fn

    raise ValueError(f"Unknown inlet_schedule type: '{stype}'")


def _build_step_fn(cfg: dict, grid: MACGrid, nu: float,
                   turb_fn, advect_fn, solve_p, apply_bc, pad_diff,
                   grad_p_fn, ibm_fn, ibm_scalar_fn=None,
                   scalar_source=None, inlet_schedule_fn=None,
                   rans_rhs_fn=None, rans_init_scalars=None):
    """Build the core time-stepping function from config."""
    from jax_cfd.core.boundaries import pad_scalar_for_diffusion as _psd

    solver   = cfg.get('solver', 'incompressible_ns')
    sch_cfg  = cfg.get('schemes', {})
    use_heat = solver in ('incompressible_ns_heat', 'incompressible_ns_heat_phase_field')
    use_phase_field = solver in ('incompressible_ns_phase_field', 'incompressible_ns_heat_phase_field')
    use_boiling = use_heat and use_phase_field
    heat_cfg = cfg.get('heat', None)
    pf_cfg   = cfg.get('phase_field', None)
    bc_cfg   = cfg.get('boundaries', {})

    # Body force (e.g. constant dp/dx for channel flow)
    bf = cfg.get('body_force', None)
    body_force = tuple(float(x) for x in bf) if bf is not None else None

    # Heat / buoyancy
    if use_heat and heat_cfg is not None:
        Pr_heat    = float(cfg.get('Pr', 0.71))
        g_vec_nd   = heat_cfg.get('g_vector', [0.0, 1.0])
        # Free-fall nondim (Re=sqrt(Ra/Pr)): buoyancy coeff = 1.0, NOT Pr
        gravity_nd = tuple(float(g) for g in g_vec_nd)
        beta_nd    = float(heat_cfg.get('beta',  1.0))
        T_ref_nd   = float(heat_cfg.get('T_ref', 0.5))
    else:
        Pr_heat    = float(cfg.get('Pr', 1.0))
        gravity_nd = (0.0,) * grid.ndim
        beta_nd    = 0.0
        T_ref_nd   = 0.0

    def _pad_scalar(s, name, g, extra=None):
        return _psd(s, name, bc_cfg, g)

    # Scalar MUSCL
    _need_scalars = use_heat or use_phase_field
    _periodic = _get_periodic_axes(bc_cfg, grid.ndim)
    if _need_scalars and sch_cfg.get('advection', 'muscl').lower() == 'muscl':
        _advect_scalar_fn = (
            partial(advect_scalar_muscl, periodic_axes=_periodic)
            if _periodic else advect_scalar_muscl
        )
    else:
        _advect_scalar_fn = None

    # Implicit diffusion (Crank-Nicolson ADI)
    _impl_diff_fn = None
    if sch_cfg.get('diffusion', 'explicit').lower() == 'implicit':
        from jax_cfd.core.implicit_diffusion import make_implicit_diffusion_fn
        _impl_diff_fn = make_implicit_diffusion_fn(
            grid, nu, pad_diff, periodic_axes=_periodic,
        )

    # Phase-field (Cahn-Hilliard) closures
    _ch_rhs_fn = None
    _momentum_src_fn = None
    if use_phase_field and pf_cfg is not None:
        from jax_cfd.core.phase_field import (
            make_ch_scalar_rhs_fn, make_surface_tension_fn, make_boiling_scalar_rhs_fn,
        )

        M_pf   = float(pf_cfg.get('mobility', 0.01))
        eps_pf = float(pf_cfg.get('epsilon', 0.05))
        sigma  = float(pf_cfg.get('sigma', 1.0))
        d_rho  = float(pf_cfg.get('delta_rho', 0.0))
        g_vec  = tuple(float(g) for g in pf_cfg.get('g_vector', [0.0, -1.0][:grid.ndim]))

        if use_boiling:
            # Build thermal body indicator fields for penalty-based CHT
            _thermal_bodies_data = None
            bodies = cfg.get('bodies', [])
            _tb_list = [b for b in bodies if 'thermal_mass' in b]
            if _tb_list:
                _thermal_bodies_data = []
                _min_dx = grid.min_spacing()
                cc = grid.cell_centers()
                for b in _tb_list:
                    btype = b.get('type', '').lower()
                    if btype in ('cylinder', 'sphere'):
                        bc = b['center']
                        R_body = float(b.get('radius', 0.5))
                        r2 = jnp.zeros(grid.cell_shape())
                        for ax in range(grid.ndim):
                            s = [1] * grid.ndim
                            s[ax] = grid.shape[ax]
                            r2 = r2 + (cc[ax].reshape(s) - float(bc[ax])) ** 2
                        r = jnp.sqrt(r2)
                        # Smooth indicator: 1 inside body, 0 outside
                        indicator = 0.5 * (1.0 - jnp.tanh((r - R_body) / (2.0 * _min_dx)))
                    else:
                        indicator = jnp.zeros(grid.cell_shape())
                    _thermal_bodies_data.append({
                        'indicator': indicator,
                        'gamma': float(b.get('gamma', 200.0)),
                        'C': float(b['thermal_mass']),
                        'Q_gen': float(b.get('Q_gen', 0.0)),
                    })

            _ch_rhs_fn = make_boiling_scalar_rhs_fn(
                grid, M_pf, eps_pf,
                _advect_scalar_fn or advect_scalar_muscl, _pad_scalar,
                Pr=Pr_heat,
                K_evap=float(pf_cfg.get('K_evap', 1.0)),
                T_sat=float(pf_cfg.get('T_sat', 0.5)),
                latent_heat=float(pf_cfg.get('latent_heat', 0.1)),
                get_nu_t_fn=turb_fn,
                thermal_bodies=_thermal_bodies_data,
            )
        else:
            _ch_rhs_fn = make_ch_scalar_rhs_fn(
                grid, M_pf, eps_pf, _advect_scalar_fn or advect_scalar_muscl, _pad_scalar,
            )
        _momentum_src_fn = make_surface_tension_fn(
            grid, sigma, eps_pf, _pad_scalar,
            delta_rho=d_rho, g_vector=g_vec,
        )

    # ── Rheology (non-Newtonian viscosity) ──────────────────────────
    _rheology_fn = None
    rheology_cfg = cfg.get('rheology', None)
    if rheology_cfg is not None:
        from jax_cfd.core.rheology import get_rheology_model
        _rheology_fn = get_rheology_model(
            rheology_cfg.get('model', 'power_law'),
            **{k: v for k, v in rheology_cfg.items() if k != 'model'},
        )

    # ── Helper: combine source functions ────────────────────────────
    def _combine_momentum_src(existing, new_src):
        """Chain two momentum_source_fn callables, summing their outputs."""
        if existing is None:
            return new_src
        def combined(scalars, velocity, grid_arg, _a=existing, _b=new_src):
            r1 = _a(scalars, velocity, grid_arg)
            r2 = _b(scalars, velocity, grid_arg)
            return tuple(a + b for a, b in zip(r1, r2))
        return combined

    def _combine_scalar_rhs(existing, new_rhs):
        """Chain two scalar_rhs_fn callables, merging their dicts."""
        if existing is None:
            return new_rhs
        def combined(scalars, vel, state, nu_val, _a=existing, _b=new_rhs):
            r1 = _a(scalars, vel, state, nu_val)
            r2 = _b(scalars, vel, state, nu_val)
            r1.update(r2)
            return r1
        return combined

    # ── Sponge layer ─────────────────────────────────────────────────
    sponge_cfg = cfg.get('sponge', None)
    if sponge_cfg is not None:
        from jax_cfd.core.sponge import make_sponge_field, make_sponge_source_fn
        _sigma, _u_tgt = make_sponge_field(grid, sponge_cfg)
        _momentum_src_fn = _combine_momentum_src(
            _momentum_src_fn, make_sponge_source_fn(grid, _sigma, _u_tgt))

    # ── MRF (moving reference frame) ─────────────────────────────────
    mrf_cfg = cfg.get('mrf', None)
    if mrf_cfg is not None:
        from jax_cfd.core.mrf import make_mrf_source_fn
        _momentum_src_fn = _combine_momentum_src(
            _momentum_src_fn,
            make_mrf_source_fn(
                grid,
                omega=float(mrf_cfg['omega']),
                center=tuple(float(c) for c in mrf_cfg.get('center', [0.5] * grid.ndim)),
            ))

    # ── Combustion (species transport + reactions) ────────────────────
    _comb_init_scalars = None
    combustion_cfg = cfg.get('combustion', None)
    if combustion_cfg is not None:
        from jax_cfd.core.combustion import make_combustion_rhs_fn
        _comb_rhs, _comb_init_scalars = make_combustion_rhs_fn(grid, combustion_cfg, nu)
        _ch_rhs_fn = _combine_scalar_rhs(_ch_rhs_fn, _comb_rhs)
        _need_scalars = True

    # ── RANS k-ω SST scalar transport ────────────────────────────────
    if rans_rhs_fn is not None:
        _ch_rhs_fn = _combine_scalar_rhs(_ch_rhs_fn, rans_rhs_fn)
        _need_scalars = True

    step_fn = create_step_fn(
        grid             = grid,
        nu_base          = nu,
        apply_bc_fn      = apply_bc,
        pad_diff_fn      = pad_diff,
        get_nu_t_fn      = turb_fn,
        advect_fn        = advect_fn,
        solve_p_fn       = solve_p,
        ibm_fn           = ibm_fn,
        time_scheme      = sch_cfg.get('time', 'rk3'),
        solve_scalars    = _need_scalars,
        solve_buoyancy   = (use_heat and 'Ra' in cfg) or use_boiling,
        gravity_vector   = gravity_nd,
        beta             = beta_nd,
        T_ref            = T_ref_nd,
        advect_scalar_fn = _advect_scalar_fn,
        pad_scalar_fn    = _pad_scalar if _need_scalars else None,
        scalar_params    = {'T': {'Pr': Pr_heat}} if use_heat else None,
        body_force       = body_force,
        gradient_p_fn    = grad_p_fn,
        ibm_scalar_fn    = ibm_scalar_fn,
        scalar_source    = scalar_source,
        implicit_diffusion_fn = _impl_diff_fn,
        scalar_rhs_fn    = _ch_rhs_fn,
        momentum_source_fn = _momentum_src_fn,
        inlet_schedule_fn = inlet_schedule_fn,
        rheology_fn      = _rheology_fn,
    )
    # Collect extra initial scalars from RANS / combustion
    _extra_init_scalars = {}
    if rans_init_scalars is not None:
        _extra_init_scalars.update(rans_init_scalars)
    if _comb_init_scalars is not None:
        _extra_init_scalars.update(_comb_init_scalars)

    return step_fn, use_heat, heat_cfg, use_phase_field, pf_cfg, _extra_init_scalars


def _parse_time_control(cfg: dict, U_ref: float, min_dx: float):
    """Parse time control config and return (max_cfl, t_end, save_every, diag_every, adaptive_dt, dt, dt_max_cfg)."""
    tc          = cfg.get('time_control', {})
    max_cfl     = float(tc.get('max_cfl',    0.4))
    t_end       = float(tc.get('t_end',      100.0))
    save_every  = int(tc.get('save_every',   200))
    diag_every  = int(tc.get('diag_every',   10))
    adaptive_dt = bool(tc.get('adaptive_dt', False))
    dt_max_cfg  = float(tc.get('dt_max', float('inf')))

    dt = min(max_cfl * min_dx / U_ref, dt_max_cfg)

    if save_every % diag_every != 0:
        save_every = (save_every // diag_every) * diag_every or diag_every

    return max_cfl, t_end, save_every, diag_every, adaptive_dt, dt, dt_max_cfg


def _parse_diagnostics(cfg: dict, ibm_fn):
    """Parse diagnostics config and return (do_forces, monitor, probe_list, probe_arr)."""
    diag_cfg  = cfg.get('diagnostics', {})
    do_forces = diag_cfg.get('forces', True)

    _default_monitor = ['Cd', 'Cl'] if ibm_fn is not None else ['ke']
    monitor: list = diag_cfg.get('monitor', _default_monitor)

    probe_list = diag_cfg.get('probes', [])
    probe_arr  = jnp.array(probe_list, dtype=jnp.float32) if probe_list else None

    return do_forces, monitor, probe_list, probe_arr


def _force_normalization(cfg: dict, rho: float, U_ref: float) -> float:
    """Return q = 0.5 * rho * U_ref² * D (reference force scale)."""
    bodies = cfg.get('bodies', [])
    if not bodies:
        return 1.0
    body  = bodies[0]
    btype = body.get('type', 'none').lower()
    if btype in ('cylinder', 'sphere'):
        D = 2.0 * float(body.get('radius', 0.5))
    elif btype == 'naca':
        D = float(body.get('chord', 1.0))
    else:
        D = float(cfg.get('reference', {}).get('L', 1.0))
    return 0.5 * rho * U_ref ** 2 * D


def _make_pressure_cd_fn(cfg: dict, grid: MACGrid, q_ref: float,
                         nu: float = None):
    """
    Build a closure that computes Cd/Cl from probe-ring extrapolation.

    Two probe rings are placed outside the IBM smearing zone (~2 dx)
    and the surface pressure (and optionally velocity gradients) are
    linearly extrapolated back to the body surface.

    When *nu* is provided the returned closure also computes viscous
    drag from the strain-rate tensor, giving a total Cd = Cd_p + Cd_v.

    Currently supports: cylinder (2D), sphere (3D).

    Returns
    -------
    cd_fn : callable or None
        ``cd_fn(pressure, velocity=None) -> dict``
        Always contains 'Cd', 'Cl' (total).
        When velocity is supplied: also 'Cd_p', 'Cl_p', 'Cd_v', 'Cl_v'.
    """
    from jax.scipy.ndimage import map_coordinates as _mc

    bodies = cfg.get('bodies', [])
    if not bodies:
        return None

    body = bodies[0]
    btype = body.get('type', '').lower()
    ndim = grid.ndim

    # ── Cylinder (2D) ────────────────────────────────────────────────
    if btype == 'cylinder' and ndim == 2:
        R = float(body.get('radius', 0.5))
        center = body.get('center', None)
        if center is None:
            cx = grid.domain_size[0] / 2.0
            cy = grid.domain_size[1] / 2.0
        else:
            cx, cy = float(center[0]), float(center[1])

        min_dx = grid.min_spacing()
        n_pts = max(int(2.0 * np.pi * R / min_dx), 64)
        theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        nx_hat = jnp.array(np.cos(theta), dtype=jnp.float32)
        ny_hat = jnp.array(np.sin(theta), dtype=jnp.float32)
        ds = 2.0 * np.pi * R / n_pts

        # Probe ring radii (outside IBM smearing zone ~2dx)
        r1 = R + 1.5 * min_dx
        r2 = R + 2.5 * min_dx
        w1 = float((R - r2) / (r1 - r2))
        w2 = float((R - r1) / (r2 - r1))

        # Pre-compute fractional cell-center indices for both rings
        xc, yc = grid.cell_centers()
        dx_c = float(xc[1] - xc[0])
        dy_c = float(yc[1] - yc[0])
        x0_c = float(xc[0])
        y0_c = float(yc[0])

        idx1 = jnp.array(np.stack([
            (cx + r1 * np.cos(theta) - x0_c) / dx_c,
            (cy + r1 * np.sin(theta) - y0_c) / dy_c,
        ]), dtype=jnp.float32)
        idx2 = jnp.array(np.stack([
            (cx + r2 * np.cos(theta) - x0_c) / dx_c,
            (cy + r2 * np.sin(theta) - y0_c) / dy_c,
        ]), dtype=jnp.float32)

        _ds_q = float(ds / q_ref)
        _nu = float(nu) if nu is not None else None
        _dx = float(grid.spacings[0])
        _dy = float(grid.spacings[1])

        def cd_fn(pressure, velocity=None):
            # ── Pressure drag ──
            p1 = _mc(pressure, idx1, order=1, mode='nearest')
            p2 = _mc(pressure, idx2, order=1, mode='nearest')
            p_surf = w1 * p1 + w2 * p2
            Cd_p = -jnp.sum(p_surf * nx_hat) * _ds_q
            Cl_p = -jnp.sum(p_surf * ny_hat) * _ds_q

            if velocity is None or _nu is None:
                return {'Cd': Cd_p, 'Cl': Cl_p}

            # ── Viscous drag (strain-rate tensor → surface traction) ──
            u, v = velocity[0], velocity[1]

            # Strain rate at cell centers (exact for MAC grid)
            dudx = (u[1:, :] - u[:-1, :]) / _dx    # (Nx, Ny)
            dvdy = (v[:, 1:] - v[:, :-1]) / _dy    # (Nx, Ny)

            # Cross-derivatives: average to cell centers, central diff
            u_cc = 0.5 * (u[:-1, :] + u[1:, :])    # (Nx, Ny)
            u_pad = jnp.pad(u_cc, ((0, 0), (1, 1)), mode='edge')
            dudy = (u_pad[:, 2:] - u_pad[:, :-2]) / (2.0 * _dy)

            v_cc = 0.5 * (v[:, :-1] + v[:, 1:])    # (Nx, Ny)
            v_pad = jnp.pad(v_cc, ((1, 1), (0, 0)), mode='edge')
            dvdx = (v_pad[2:, :] - v_pad[:-2, :]) / (2.0 * _dx)

            # Interpolate to probe rings & extrapolate to surface
            dudx_s = w1 * _mc(dudx, idx1, order=1, mode='nearest') \
                   + w2 * _mc(dudx, idx2, order=1, mode='nearest')
            dvdy_s = w1 * _mc(dvdy, idx1, order=1, mode='nearest') \
                   + w2 * _mc(dvdy, idx2, order=1, mode='nearest')
            dudy_s = w1 * _mc(dudy, idx1, order=1, mode='nearest') \
                   + w2 * _mc(dudy, idx2, order=1, mode='nearest')
            dvdx_s = w1 * _mc(dvdx, idx1, order=1, mode='nearest') \
                   + w2 * _mc(dvdx, idx2, order=1, mode='nearest')

            # Viscous traction: t_i = τ_ij n_j = μ(∂u_i/∂x_j + ∂u_j/∂x_i) n_j
            tx = _nu * (2.0 * dudx_s * nx_hat + (dudy_s + dvdx_s) * ny_hat)
            ty = _nu * ((dudy_s + dvdx_s) * nx_hat + 2.0 * dvdy_s * ny_hat)
            Cd_v = jnp.sum(tx) * _ds_q
            Cl_v = jnp.sum(ty) * _ds_q

            return {
                'Cd': Cd_p + Cd_v, 'Cl': Cl_p + Cl_v,
                'Cd_p': Cd_p, 'Cl_p': Cl_p,
                'Cd_v': Cd_v, 'Cl_v': Cl_v,
            }

        return cd_fn

    # ── Sphere (3D) ──────────────────────────────────────────────────
    if btype == 'sphere' and ndim == 3:
        R = float(body.get('radius', 0.5))
        center = body.get('center', None)
        if center is None:
            cx = grid.domain_size[0] / 2.0
            cy = grid.domain_size[1] / 2.0
            cz = grid.domain_size[2] / 2.0
        else:
            cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

        min_dx = grid.min_spacing()
        n_pts = max(int(4.0 * np.pi * R ** 2 / min_dx ** 2), 128)
        golden = (1 + np.sqrt(5)) / 2.0
        indices = np.arange(n_pts)
        theta = np.arccos(1 - 2 * (indices + 0.5) / n_pts)
        phi = 2.0 * np.pi * indices / golden

        _nx = np.sin(theta) * np.cos(phi)
        _ny = np.sin(theta) * np.sin(phi)
        _nz = np.cos(theta)
        nx_hat = jnp.array(_nx, dtype=jnp.float32)
        ny_hat = jnp.array(_ny, dtype=jnp.float32)
        nz_hat = jnp.array(_nz, dtype=jnp.float32)
        ds = 4.0 * np.pi * R ** 2 / n_pts

        r1 = R + 1.5 * min_dx
        r2 = R + 2.5 * min_dx
        w1 = float((R - r2) / (r1 - r2))
        w2 = float((R - r1) / (r2 - r1))

        xc, yc, zc = grid.cell_centers()
        dx_c = float(xc[1] - xc[0])
        dy_c = float(yc[1] - yc[0])
        dz_c = float(zc[1] - zc[0])
        x0_c, y0_c, z0_c = float(xc[0]), float(yc[0]), float(zc[0])

        idx1 = jnp.array(np.stack([
            (cx + r1 * _nx - x0_c) / dx_c,
            (cy + r1 * _ny - y0_c) / dy_c,
            (cz + r1 * _nz - z0_c) / dz_c,
        ]), dtype=jnp.float32)
        idx2 = jnp.array(np.stack([
            (cx + r2 * _nx - x0_c) / dx_c,
            (cy + r2 * _ny - y0_c) / dy_c,
            (cz + r2 * _nz - z0_c) / dz_c,
        ]), dtype=jnp.float32)

        _ds_q = float(ds / q_ref)

        def cd_fn(pressure, velocity=None):
            p1 = _mc(pressure, idx1, order=1, mode='nearest')
            p2 = _mc(pressure, idx2, order=1, mode='nearest')
            p_surf = w1 * p1 + w2 * p2
            Cd = -jnp.sum(p_surf * nx_hat) * _ds_q
            Cl = -jnp.sum(p_surf * ny_hat) * _ds_q
            Cz = -jnp.sum(p_surf * nz_hat) * _ds_q
            return {'Cd': Cd, 'Cl': Cl, 'Cz': Cz}

        return cd_fn

    return None


def _init_state(
    cfg: dict,
    grid: MACGrid,
    has_ibm: bool = False,
    heat_cfg: dict = None,
    pf_cfg: dict = None,
    inlet_vel=None,
    extra_init_scalars: dict = None,
) -> FluidState:
    ic     = cfg.get('initial_conditions', {})
    bc_cfg = cfg.get('boundaries', {})
    ndim   = grid.ndim

    vel_type = ic.get('velocity_type', 'uniform')
    if vel_type == 'shear':
        # Tanh shear profile: u = U_shear * tanh((y - y_mid) / delta)
        U_sh   = float(ic.get('shear_velocity', 0.5))
        sh_ax  = int(ic.get('shear_axis', ndim - 1))
        sh_pos = float(ic.get('shear_pos', grid.domain_size[sh_ax] / 2))
        sh_w   = float(ic.get('shear_width', 0.05))
        flow_ax = int(ic.get('shear_flow_axis', 0))
        vel_list = []
        for i in range(ndim):
            if i == flow_ax:
                # Face centres along flow_ax for the u-component
                fc = grid.face_centers(i)
                s = [1] * ndim
                s[sh_ax] = fc[sh_ax].shape[0]
                y_face = fc[sh_ax].reshape(s)
                u_profile = U_sh * jnp.tanh((y_face - sh_pos) / sh_w)
                vel_list.append(jnp.broadcast_to(u_profile, grid.face_shape(i)))
            else:
                vel_list.append(jnp.zeros(grid.face_shape(i)))
        vel = tuple(vel_list)
    else:
        vel_ic = ic.get('velocity', [1.0] + [0.0] * (ndim - 1))
        vel = tuple(jnp.full(grid.face_shape(i), float(vel_ic[i])) for i in range(ndim))
    vel = apply_boundaries(vel, bc_cfg, grid, inlet_vel=inlet_vel)
    p   = jnp.zeros(grid.cell_shape())

    ibm_forces = jnp.zeros(ndim) if has_ibm else None

    if heat_cfg is not None:
        ic_type = heat_cfg.get('ic_type', 'linear')

        if ic_type == 'gaussian':
            cc = grid.cell_centers()
            center = heat_cfg.get('center',
                                  [float(L) / 2 for L in grid.domain_size])
            sigma = float(heat_cfg.get('sigma', 0.5))
            amplitude = float(heat_cfg.get('amplitude', 1.0))
            r2 = jnp.zeros(grid.cell_shape())
            for i in range(ndim):
                s = [1] * ndim
                s[i] = grid.shape[i]
                r2 = r2 + (cc[i].reshape(s) - float(center[i])) ** 2
            scalars = {'T': amplitude * jnp.exp(-r2 / (2.0 * sigma ** 2))}

        elif ic_type == 'uniform':
            T_val = float(heat_cfg.get('T_value', 0.0))
            scalars = {'T': jnp.full(grid.cell_shape(), T_val)}

        else:  # 'linear' — default (Rayleigh-Bénard)
            T_hot  = float(heat_cfg.get('T_hot',  1.0))
            T_cold = float(heat_cfg.get('T_cold', 0.0))
            amp    = float(heat_cfg.get('perturb_amp', 0.01))
            wall_axis = int(heat_cfg.get('wall_axis', ndim - 1))
            cc = grid.cell_centers()
            L_wall = grid.domain_size[wall_axis]
            y_1d = cc[wall_axis]

            T_lin_shape = [1] * ndim
            T_lin_shape[wall_axis] = grid.shape[wall_axis]
            T_init = jnp.broadcast_to(
                (T_hot + (T_cold - T_hot) * y_1d / L_wall).reshape(T_lin_shape),
                grid.cell_shape(),
            )

            other_ax = [i for i in range(ndim) if i != wall_axis][0]
            L_other = grid.domain_size[other_ax]
            x_1d = cc[other_ax]
            y_shape = [1] * ndim
            y_shape[wall_axis] = grid.shape[wall_axis]
            x_shape = [1] * ndim
            x_shape[other_ax] = grid.shape[other_ax]
            perturb = amp * (
                jnp.sin(jnp.pi * y_1d.reshape(y_shape) / L_wall)
                * jnp.cos(2.0 * jnp.pi * x_1d.reshape(x_shape) / L_other))
            scalars = {'T': T_init + perturb}
    else:
        scalars = {}

    # Phase-field initial condition
    if pf_cfg is not None:
        eps_pf = float(pf_cfg.get('epsilon', 0.05))
        ic_pf = pf_cfg.get('ic_type', 'bubble')
        cc = grid.cell_centers()

        if ic_pf == 'bubble':
            center_pf = pf_cfg.get('center',
                                   [float(L) / 2 for L in grid.domain_size])
            radius_pf = float(pf_cfg.get('radius', 0.25))
            r2 = jnp.zeros(grid.cell_shape())
            for i in range(ndim):
                s = [1] * ndim
                s[i] = grid.shape[i]
                r2 = r2 + (cc[i].reshape(s) - float(center_pf[i])) ** 2
            r = jnp.sqrt(r2)
            # φ = -tanh((r-R)/(√2 ε)):  +1 inside bubble, -1 outside
            scalars['phi'] = -jnp.tanh((r - radius_pf) / (jnp.sqrt(2.0) * eps_pf))

        elif ic_pf == 'random':
            amp_pf = float(pf_cfg.get('perturb_amp', 0.05))
            mean_pf = float(pf_cfg.get('mean_phi', 0.0))
            key = jax.random.PRNGKey(int(pf_cfg.get('seed', 42)))
            scalars['phi'] = mean_pf + amp_pf * jax.random.normal(key, grid.cell_shape())

        elif ic_pf == 'layer':
            # Horizontal interface: φ = tanh((y - y_intf)/(√2 ε))
            # Optional sinusoidal perturbation for Rayleigh-Taylor instability.
            axis_pf = int(pf_cfg.get('interface_axis', ndim - 1))
            y_mid = float(pf_cfg.get('interface_pos',
                                     grid.domain_size[axis_pf] / 2))
            perturb_amp_pf = float(pf_cfg.get('perturb_amp', 0.0))
            perturb_modes = int(pf_cfg.get('perturb_modes', 1))

            s_pf = [1] * ndim
            s_pf[axis_pf] = grid.shape[axis_pf]
            y_1d = cc[axis_pf].reshape(s_pf)

            # Build interface position (possibly x-dependent perturbation)
            y_intf = y_mid
            if abs(perturb_amp_pf) > 1e-12 and ndim >= 2:
                # Perturbation along the first axis that is NOT interface_axis
                perturb_axis = 0 if axis_pf != 0 else 1
                Lp = grid.domain_size[perturb_axis]
                sx = [1] * ndim
                sx[perturb_axis] = grid.shape[perturb_axis]
                x_1d = cc[perturb_axis].reshape(sx)
                y_intf = y_mid
                for mode in range(1, perturb_modes + 1):
                    phase = 0.3 * (mode - 1)  # offset phases
                    amp_mode = perturb_amp_pf / mode  # decreasing amplitude
                    y_intf = y_intf + amp_mode * jnp.cos(
                        2.0 * jnp.pi * mode * x_1d / Lp + phase
                    )

            scalars['phi'] = jnp.tanh(
                (y_1d - y_intf) / (jnp.sqrt(2.0) * eps_pf)
            )

        elif ic_pf == 'boiling':
            # Thin vapor layer at bottom with perturbation → nucleation
            y_layer = float(pf_cfg.get('vapor_layer_height', 0.15))
            perturb_amp_pf = float(pf_cfg.get('perturb_amp', 0.05))
            y_axis = ndim - 1
            Lx_pf = grid.domain_size[0]

            y_shape = [1] * ndim
            y_shape[y_axis] = grid.shape[y_axis]
            y_1d = cc[y_axis].reshape(y_shape)

            x_shape = [1] * ndim
            x_shape[0] = grid.shape[0]
            x_1d = cc[0].reshape(x_shape)

            y_interface = y_layer + perturb_amp_pf * (
                jnp.sin(2.0 * jnp.pi * x_1d / Lx_pf)
                + 0.5 * jnp.sin(4.0 * jnp.pi * x_1d / Lx_pf + 0.3)
            )
            # Vapor (φ=+1) below interface, liquid (φ=-1) above
            scalars['phi'] = -jnp.tanh(
                (y_1d - y_interface) / (jnp.sqrt(2.0) * eps_pf)
            )

        elif ic_pf == 'body_nucleation':
            # Thin vapor shell around each IBM body for nucleate boiling.
            # φ=+1 (vapor) near body surface, φ=-1 (liquid) far away.
            vapor_thickness = float(pf_cfg.get('vapor_thickness', 0.1))
            phi_init = -jnp.ones(grid.cell_shape())
            bodies = cfg.get('bodies', [])
            sqrt2_eps = jnp.sqrt(2.0) * eps_pf
            for body in bodies:
                btype = body.get('type', '').lower()
                if btype in ('cylinder', 'sphere'):
                    bc = body['center']
                    R_body = float(body.get('radius', 0.5))
                    r2 = jnp.zeros(grid.cell_shape())
                    for i in range(ndim):
                        s = [1] * ndim
                        s[i] = grid.shape[i]
                        r2 = r2 + (cc[i].reshape(s) - float(bc[i])) ** 2
                    r = jnp.sqrt(r2)
                    # Transition at R + vapor_thickness/2
                    r_mid = R_body + vapor_thickness / 2.0
                    phi_body = -jnp.tanh((r - r_mid) / sqrt2_eps)
                    phi_init = jnp.maximum(phi_init, phi_body)
            # Small noise to break azimuthal symmetry
            perturb_amp_pf = float(pf_cfg.get('perturb_amp', 0.02))
            if perturb_amp_pf > 0:
                key = jax.random.PRNGKey(int(pf_cfg.get('seed', 42)))
                phi_init = phi_init + perturb_amp_pf * jax.random.normal(
                    key, grid.cell_shape())
                phi_init = jnp.clip(phi_init, -1.0, 1.0)
            scalars['phi'] = phi_init

        else:
            scalars['phi'] = jnp.zeros(grid.cell_shape())

    # Initialize body_temps for thermal bodies (penalty-based CHT)
    bodies = cfg.get('bodies', [])
    _tb_list = [b for b in bodies if 'thermal_mass' in b]
    if _tb_list:
        T_init_bodies = [float(b.get('T', 1.0)) for b in _tb_list]
        scalars['body_temps'] = jnp.array(T_init_bodies)

    # Merge extra initial scalars (RANS k/omega, combustion species, etc.)
    if extra_init_scalars:
        for name, val in extra_init_scalars.items():
            if name not in scalars:
                scalars[name] = val

    return FluidState(velocity=vel, pressure=p, scalars=scalars, ibm_forces=ibm_forces)


def _load_restart_state(
    snap: dict,
    grid: MACGrid,
    has_ibm: bool,
    use_heat: bool,
    use_phase_field: bool = False,
) -> FluidState:
    """Reconstruct a FluidState from an HDF5 snapshot dict."""
    comp_names = ['u', 'v', 'w'][:grid.ndim]
    vel = tuple(jnp.array(snap[n]) for n in comp_names)
    p = jnp.array(snap['p'])
    scalars = {}
    if use_heat and 'T' in snap:
        scalars['T'] = jnp.array(snap['T'])
    if use_phase_field and 'phi' in snap:
        scalars['phi'] = jnp.array(snap['phi'])
    if 'body_temps' in snap:
        scalars['body_temps'] = jnp.array(snap['body_temps'])
    ibm_forces = jnp.zeros(grid.ndim) if has_ibm else None
    return FluidState(velocity=vel, pressure=p, scalars=scalars, ibm_forces=ibm_forces)


def _run_compressible(cfg: dict) -> FluidState:
    """Runner for the compressible Euler/NS solver.

    The compressible solver has its own step function (density-based, no
    pressure Poisson) so it bypasses the incompressible build chain entirely.
    Supports CFL-based adaptive dt and HDF5 output.
    """
    from jax_cfd.core.compressible import (
        make_compressible_step_fn, sod_shock_tube_ic,
        explosion_ic, mach_flow_ic, sound_speed,
    )
    import h5py

    grid = _build_grid(cfg)
    gamma = float(cfg.get('gamma', 1.4))
    solver = cfg.get('solver', 'compressible_euler')
    mu = float(cfg.get('mu', 0.0)) if solver == 'compressible_ns' else 0.0

    step_fn = make_compressible_step_fn(grid, gamma, mu)

    # ── Initial condition ──────────────────────────────────────
    ic_cfg = cfg.get('initial_conditions', {})
    ic_type = ic_cfg.get('type', 'sod')

    def _make_ic():
        if ic_type == 'sod':
            return sod_shock_tube_ic(grid, gamma)
        elif ic_type == 'explosion':
            return explosion_ic(
                grid, gamma,
                center=ic_cfg.get('center'),
                radius=float(ic_cfg.get('radius', 0.1)),
                p_high=float(ic_cfg.get('p_high', 100.0)),
                p_low=float(ic_cfg.get('p_low', 1.0)),
            )
        elif ic_type == 'mach_flow':
            return mach_flow_ic(
                grid, gamma,
                Mach=float(ic_cfg.get('Mach', 2.0)),
            )
        else:
            raise ValueError(f"Unknown compressible IC type: {ic_type}")

    state = _make_ic()

    # ── Time control ───────────────────────────────────────────
    tc = cfg.get('time_control', {})
    t_end = float(tc.get('t_end', 0.2))
    max_cfl = float(tc.get('max_cfl', 0.4))
    adaptive = tc.get('adaptive_dt', True)
    dt_fixed = float(tc.get('dt', 1e-4))
    diag_every = int(tc.get('diag_every', 100))
    save_every = int(tc.get('save_every', 200))

    # ── Output ─────────────────────────────────────────────────
    out_cfg = cfg.get('output', {})
    out_dir = Path(out_cfg.get('dir', 'output'))
    out_file = out_cfg.get('file', 'run.h5')
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / out_file

    _p = lambda *a, **kw: print(*a, **kw, flush=True)
    _p(f"[Runner] solver   = {solver}  (compressible)")
    _p(f"[Runner] grid     = {grid.shape}  ({grid.domain_size})")
    _p(f"[Runner] gamma    = {gamma},  mu = {mu},  IC = {ic_type}")
    _p(f"[Runner] CFL      = {max_cfl}  (adaptive={adaptive})")
    _p(f"[Runner] output   → {h5_path}")

    # ── CFL-based dt ───────────────────────────────────────────
    def _compute_dt(st):
        rho = st.scalars['rho']
        p = st.pressure
        c = sound_speed(rho, p, gamma)
        max_wave = 0.0
        for i in range(grid.ndim):
            max_wave = jnp.maximum(max_wave,
                                   jnp.max(jnp.abs(st.velocity[i]) + c))
        dx_min = min(grid.spacings)
        return max_cfl * dx_min / jnp.maximum(max_wave, 1e-14)

    # ── JIT warm-up ────────────────────────────────────────────
    _p("[Runner] JIT warm-up (compressible)...")
    t_jit = time.time()
    _dt0 = _compute_dt(state) if adaptive else dt_fixed
    _dummy = step_fn(state, float(_dt0), 0.0)
    jax.block_until_ready(_dummy.velocity)
    _p(f"[Runner] JIT done in {time.time() - t_jit:.1f} s\n")

    # Re-init after warm-up (JIT can corrupt state in some cases)
    state = _make_ic()

    # ── Write HDF5 header ──────────────────────────────────────
    h5 = h5py.File(str(h5_path), 'w')
    h5.attrs['solver'] = solver
    h5.attrs['gamma'] = gamma
    h5.attrs['ic_type'] = ic_type
    grp_grid = h5.create_group('grid')
    grp_grid.attrs['ndim'] = grid.ndim
    cc = grid.cell_centers()
    axis_names = ['xc', 'yc', 'zc']
    for i in range(grid.ndim):
        grp_grid.create_dataset(axis_names[i], data=np.array(cc[i]))
    h5.create_group('fields')
    h5.flush()

    def _save_snapshot(st, step_i, t_i):
        grp_name = f'fields/step_{step_i:07d}'
        g = h5.create_group(grp_name)
        g.attrs['t'] = t_i
        g.attrs['step'] = step_i
        opts = dict(compression='gzip', compression_opts=4, dtype=np.float32)
        comp_names = ['u', 'v', 'w']
        for i in range(grid.ndim):
            g.create_dataset(comp_names[i], data=np.array(st.velocity[i]), **opts)
        g.create_dataset('p', data=np.array(st.pressure), **opts)
        g.create_dataset('rho', data=np.array(st.scalars['rho']), **opts)
        g.create_dataset('E', data=np.array(st.scalars['E']), **opts)
        h5.flush()

    # Save initial state
    _save_snapshot(state, 0, 0.0)

    # ── Main time loop ─────────────────────────────────────────
    _p(f"{'step':>8s}  {'t':>10s}  {'dt':>10s}  {'rho_min':>8s}  {'rho_max':>8s}  "
       f"{'p_max':>8s}  {'Mach_max':>8s}  {'wall(s)':>8s}")
    _p('-' * 80)

    t_sim = 0.0
    step_count = 0
    t_wall = time.time()

    while t_sim < t_end - 1e-14:
        dt = float(_compute_dt(state)) if adaptive else dt_fixed
        # Don't overshoot t_end
        dt = min(dt, t_end - t_sim)

        state = step_fn(state, dt, t_sim)
        t_sim += dt
        step_count += 1

        if step_count % diag_every == 0:
            rho = state.scalars['rho']
            p = state.pressure
            c = sound_speed(rho, p, gamma)
            speed = jnp.sqrt(sum(v ** 2 for v in state.velocity))
            mach = speed / jnp.maximum(c, 1e-14)
            elapsed = time.time() - t_wall
            _p(f"  {step_count:6d}  {t_sim:10.5f}  {dt:10.6f}  "
               f"{float(jnp.min(rho)):8.4f}  {float(jnp.max(rho)):8.4f}  "
               f"{float(jnp.max(p)):8.2f}  {float(jnp.max(mach)):8.3f}  "
               f"{elapsed:8.1f}")

        if step_count % save_every == 0:
            _save_snapshot(state, step_count, t_sim)

    # Save final snapshot
    _save_snapshot(state, step_count, t_sim)
    h5.close()

    _p(f"\n[Runner] Compressible finished at t={t_sim:.5f}, step={step_count}")
    _p(f"[Runner] Output: {h5_path}")
    return state


# ── Main entry point ───────────────────────────────────────────────────────

def run(config: dict, restart_from: Optional[str] = None, callbacks=None,
        dry_run: bool = False) -> Optional[FluidState]:
    """
    Run a simulation defined by *config* and write output to HDF5.

    Parameters
    ----------
    config : dict
        Merged config dict (DOMAIN | PHYSICS | CONTROL).
    restart_from : str, optional
        Path to an existing HDF5 file to restart from.  The last saved
        snapshot is used as the initial condition and the simulation
        continues from that time.  If the path matches the output path,
        new data is appended to the same file; otherwise a new output
        file is created.
    callbacks : list of callable, optional
        Functions called every ``diag_every`` steps with keyword arguments
        ``step``, ``t``, ``state``, ``monitor``, ``grid``, ``cfl``, ``dt``.
        Exceptions inside callbacks are silently caught so they never
        abort the simulation.
    dry_run : bool
        If True, print simulation parameters and estimated resource usage
        without running.  Returns None.

    Returns
    -------
    Final FluidState (or None if dry_run=True).
    """
    callbacks = callbacks or []
    cfg = dict(config)

    # Auto-derive Re = sqrt(Ra/Pr) for the heat solver
    if cfg.get('solver') in ('incompressible_ns_heat', 'incompressible_ns_heat_phase_field') \
       and 'Ra' in cfg and 'Re' not in cfg:
        _Ra = float(cfg['Ra'])
        _Pr = float(cfg.get('Pr', 0.71))
        cfg['Re'] = float(np.sqrt(_Ra / _Pr))

    _validate(cfg)

    # ── Compressible solver: separate code path ──────────────────────
    solver = cfg.get('solver', 'incompressible_ns')
    if solver.startswith('compressible'):
        return _run_compressible(cfg)

    # ── Build components ──────────────────────────────────────────────
    grid                   = _build_grid(cfg)
    nu, U_ref, L_ref, rho  = _build_physics_params(cfg)
    turb_fn, advect_fn, solve_p, apply_bc, pad_diff, grad_p_fn, rans_rhs_fn, rans_init_scalars = _build_closures(cfg, grid)
    ibm_fn, coords_list, ds_list = _build_ibm(cfg, grid)
    ibm_scalar_fn, scalar_source = _build_ibm_scalar(cfg, grid)

    solver = cfg.get('solver', 'incompressible_ns')
    inlet_schedule_fn = _build_inlet_schedule(cfg, grid)
    step_fn, use_heat, heat_cfg, use_phase_field, pf_cfg, extra_init_scalars = _build_step_fn(
        cfg, grid, nu, turb_fn, advect_fn, solve_p, apply_bc, pad_diff,
        grad_p_fn, ibm_fn, ibm_scalar_fn, scalar_source, inlet_schedule_fn,
        rans_rhs_fn, rans_init_scalars,
    )

    # ── Time control ──────────────────────────────────────────────────
    min_dx = grid.min_spacing()
    max_cfl, t_end, save_every, diag_every, adaptive_dt, dt, dt_max_cfg = \
        _parse_time_control(cfg, U_ref, min_dx)
    max_dt = min(dt, dt_max_cfg)
    q_ref  = _force_normalization(cfg, rho, U_ref)

    # ── Pressure+viscous Cd/Cl closure (replaces O(1/dt) IBM forces) ──
    pressure_cd_fn = _make_pressure_cd_fn(cfg, grid, q_ref, nu=nu)

    # ── Diagnostics config ────────────────────────────────────────────
    do_forces, monitor, probe_list, probe_arr = _parse_diagnostics(cfg, ibm_fn)

    # ── Output ────────────────────────────────────────────────────────
    out_cfg  = cfg.get('output', {})
    out_dir  = Path(out_cfg.get('dir',  'output'))
    out_file = out_cfg.get('file', 'run.h5')
    h5_path  = out_dir / out_file

    # ── Restart setup ─────────────────────────────────────────────────
    t_start    = 0.0
    step_start = 0
    rst_snap   = None

    if restart_from is not None:
        from jax_cfd.io.reader import load_snapshot, list_steps as _list_steps, load_grid as _load_grid
        if not Path(restart_from).exists():
            raise FileNotFoundError(f"Restart file not found: {restart_from}")
        rst_steps = _list_steps(restart_from)
        if not rst_steps:
            raise ValueError(f"No snapshots found in restart file: {restart_from}")
        rst_snap   = load_snapshot(restart_from, rst_steps[-1])
        t_start    = float(rst_snap['t'])
        step_start = int(rst_snap['step'])

        # Validate grid compatibility
        rst_grid = _load_grid(restart_from)
        rst_ndim = len([k for k in rst_grid if k.endswith('c')])  # xc,yc,[zc]
        if rst_ndim != grid.ndim:
            raise ValueError(
                f"Restart file is {rst_ndim}D but current config is {grid.ndim}D."
            )
        # Check shape matches
        axis_names = ['x', 'y', 'z'][:grid.ndim]
        for ax_i, ax_name in enumerate(axis_names):
            key = f'{ax_name}c'
            if key in rst_grid:
                rst_n = len(rst_grid[key])
                cfg_n = grid.shape[ax_i]
                if rst_n != cfg_n:
                    raise ValueError(
                        f"Grid shape mismatch on {ax_name}-axis: "
                        f"restart file has {rst_n} cells, config has {cfg_n}."
                    )

    # Whether to append to the same file or start fresh
    restart_append = (
        restart_from is not None
        and Path(restart_from).resolve() == h5_path.resolve()
    )

    # ── JIT-compiled chunk runner (dt as traced argument) ─────────────
    # Passing dt as a runtime argument avoids retracing when adaptive_dt
    # changes the time step between chunks.
    _sfn = step_fn

    @jax.jit(donate_argnums=(0,))
    def run_chunk(state: FluidState, t0: jnp.ndarray, dt_arg: jnp.ndarray) -> FluidState:
        def body(carry, _):
            s, t = carry
            s = _sfn(s, dt_arg, t)
            return (s, t + dt_arg), None
        (s_final, _), _ = jax.lax.scan(body, (state, t0), length=diag_every)
        return s_final

    # ── Initial state ─────────────────────────────────────────────────
    _has_ibm = ibm_fn is not None
    _inlet_vel_init = None
    if inlet_schedule_fn is not None:
        _t_init = jnp.float32(t_start)
        _inlet_vel_init = inlet_schedule_fn(_t_init)
    if rst_snap is not None:
        state = _load_restart_state(rst_snap, grid, _has_ibm, use_heat, use_phase_field)
    else:
        state = _init_state(cfg, grid, has_ibm=_has_ibm, heat_cfg=heat_cfg, pf_cfg=pf_cfg,
                            inlet_vel=_inlet_vel_init, extra_init_scalars=extra_init_scalars)

    # ── Print summary ─────────────────────────────────────────────────
    # flush=True on all prints so subprocess stdout is unbuffered for GUI
    _p = lambda *a, **kw: print(*a, **kw, flush=True)
    _p()
    _p(f"[Runner] solver   = {solver}")
    _grid_info = f"{grid.shape}  ({grid.domain_size})"
    if not grid.is_uniform:
        _s = grid.stretch
        _grid_info += f"  stretch: axis={_s['axis']}, type={_s.get('type','tanh')}, ratio={_s.get('ratio',1)}"
    _p(f"[Runner] grid     = {_grid_info}")
    _p(f"[Runner] Re       = {cfg['Re']:.4g}   nu = {nu:.3e}")
    if use_heat:
        _p(f"[Runner] Ra       = {cfg.get('Ra', 'N/A')}   Pr = {cfg['Pr']}")
    if use_phase_field and pf_cfg is not None:
        _p(f"[Runner] phase-field: M={pf_cfg.get('mobility')}, eps={pf_cfg.get('epsilon')}, sigma={pf_cfg.get('sigma')}")
    if use_heat and use_phase_field and pf_cfg is not None:
        _p(f"[Runner] boiling: K_evap={pf_cfg.get('K_evap')}, T_sat={pf_cfg.get('T_sat')}, latent_heat={pf_cfg.get('latent_heat')}")
    _tb_list_run = [b for b in cfg.get('bodies', []) if 'thermal_mass' in b]
    if _tb_list_run:
        for _bi, _tb in enumerate(_tb_list_run):
            _p(f"[Runner] thermal body[{_bi}]: C={_tb['thermal_mass']}, Q_gen={_tb.get('Q_gen',0)}, gamma={_tb.get('gamma',200)}")
    _p(f"[Runner] dt       = {dt:.5f}  {'(adaptive)' if adaptive_dt else '(fixed)'}")
    if pressure_cd_fn is not None:
        _p(f"[Runner] Cd/Cl    = pressure-integration (probe-ring extrapolation)")
    elif ibm_fn is not None:
        _p(f"[Runner] Cd/Cl    = IBM direct-forcing (approximate)")
    if inlet_schedule_fn is not None:
        _sch = cfg.get('inlet_schedule', {})
        _p(f"[Runner] inlet    = schedule({_sch.get('type','ramp')}): "
           f"off={_sch.get('velocity_off')}, on={_sch.get('velocity_on')}, "
           f"t_on={_sch.get('t_on')}, t_off={_sch.get('t_off')}")
    if restart_from is not None:
        _p(f"[Runner] restart  → t={t_start:.3f}, step={step_start}")
    _p(f"[Runner] output   → {h5_path}")
    _p()

    # ── Dry-run: print estimates and return ───────────────────────────
    if dry_run:
        t_remain = t_end - t_start
        total_steps = int(np.ceil(t_remain / dt))
        total_chunks = int(np.ceil(total_steps / diag_every))
        n_snapshots = int(np.ceil(total_steps / save_every))
        # Estimate snapshot size from initial state arrays
        _nbytes = sum(u.nbytes for u in state.velocity) + state.pressure.nbytes
        for s in state.scalars.values():
            _nbytes += s.nbytes
        snap_mb = _nbytes / (1024 * 1024)
        total_mb = snap_mb * n_snapshots

        _p("─" * 60)
        _p("  DRY-RUN ESTIMATE")
        _p("─" * 60)
        _p(f"  Time range       : {t_start:.3f} → {t_end:.3f}  (Δt_sim = {t_remain:.3f})")
        _p(f"  dt               : {dt:.5f}")
        _p(f"  Total steps      : {total_steps:,}")
        _p(f"  Diag chunks      : {total_chunks:,}  (every {diag_every} steps)")
        _p(f"  Snapshots        : {n_snapshots:,}  (every {save_every} steps)")
        _p(f"  Snapshot size    : {snap_mb:.1f} MB each")
        _p(f"  Est. disk usage  : {total_mb:.0f} MB  ({total_mb/1024:.1f} GB)")
        _p(f"  Grid DOFs        : {int(np.prod(grid.shape)):,} cells")
        if _has_ibm:
            n_lag = sum(c[0].shape[0] for c in coords_list)
            _p(f"  IBM markers      : {n_lag:,}")
        _p("─" * 60)
        _p()
        return None

    # ── JIT warm-up ───────────────────────────────────────────────────
    _p(f"[Runner] JIT warm-up... (cache: {_cache_dir})")
    t_jit = time.time()
    _dummy = run_chunk(state, jnp.float32(t_start), jnp.float32(dt))
    jax.block_until_ready(_dummy.velocity)
    _jit_sec = time.time() - t_jit
    _jit_tag = "cached" if _jit_sec < 5.0 else "compiled"
    _p(f"[Runner] JIT done in {_jit_sec:.1f} s ({_jit_tag})\n")

    # Reset state after warm-up
    if rst_snap is not None:
        state = _load_restart_state(rst_snap, grid, _has_ibm, use_heat, use_phase_field)
    else:
        state = _init_state(cfg, grid, has_ibm=_has_ibm, heat_cfg=heat_cfg, pf_cfg=pf_cfg,
                            inlet_vel=_inlet_vel_init, extra_init_scalars=extra_init_scalars)

    # ── Progress header ───────────────────────────────────────────────
    mon_header = ''.join(f'  {q:>9}' for q in monitor)
    header = f"{'step':>8}  {'t':>8}{mon_header}  {'CFL':>5}  {'%':>5}  {'wall(s)':>7}"
    _p(header)
    _p('-' * len(header))

    # ── Strouhal number estimation (online FFT of Cl history) ────────
    _st_enabled = 'Cl' in monitor and ibm_fn is not None
    _cl_history_t = []   # time stamps
    _cl_history_v = []   # Cl values
    _st_value = float('nan')
    _st_D = 2.0 * float(cfg.get('bodies', [{}])[0].get('radius', 0.5)) if _st_enabled else 1.0

    # ── Early stopping (convergence detection) ────────────────────────
    _conv_cfg = cfg.get('convergence', {})
    _conv_enabled = bool(_conv_cfg)
    _conv_tol = float(_conv_cfg.get('tol', 1e-4))
    _conv_qty = _conv_cfg.get('quantity', 'Cd')
    _conv_window = int(_conv_cfg.get('window', 20))
    _conv_history = []
    _converged = False

    # ── Main loop ─────────────────────────────────────────────────────
    t_sim      = jnp.float32(t_start)
    step       = step_start
    chunk_i    = 0
    mon_vals: dict = {q: float('nan') for q in monitor}

    with H5Writer(str(h5_path)) as writer:
        writer.open(cfg, grid, append=restart_append)
        t_wall_start = time.time()

        while float(t_sim) < t_end - 0.5 * dt:

            # Adaptive dt: recompute from previous chunk's max velocity
            # u_max is computed once per chunk (below) and reused here.
            if adaptive_dt and chunk_i > 0:
                dt = min(max_cfl * min_dx / max(u_max, 1e-10), max_dt)

            state  = run_chunk(state, t_sim, jnp.float32(dt))
            t_sim  = t_sim + diag_every * dt
            step   = step_start + (chunk_i + 1) * diag_every
            chunk_i += 1

            # ── Batched on-device computation ──────────────────────────
            # Compute u_max, monitor quantities, and probes as JAX arrays
            # then transfer everything to host in a single device_get().
            u_max_jax = jnp.max(jnp.stack(
                [jnp.max(jnp.abs(u)) for u in state.velocity]
            ))

            # Monitor quantities (returns dict of host floats, single transfer)
            mon_vals = _compute_monitor(monitor, state, grid, q_ref, heat_cfg=heat_cfg)

            # Track body temperatures in monitor
            if 'body_temps' in state.scalars:
                _bt = jax.device_get(state.scalars['body_temps'])
                for _bi in range(len(_bt)):
                    mon_vals[f'T_body_{_bi}'] = float(_bt[_bi])

            # Pressure+viscous Cd/Cl (accurate, replaces O(1/dt) IBM forces)
            _pcd_host = {}
            if pressure_cd_fn is not None:
                _pcd = pressure_cd_fn(state.pressure, state.velocity)
                _pcd_host = {k: float(v) for k, v in jax.device_get(_pcd).items()}
                for _k in ('Cd', 'Cl', 'Cz'):
                    if _k in _pcd_host and _k in mon_vals:
                        mon_vals[_k] = _pcd_host[_k]

            # u_max: single device→host sync (piggyback on block_until_ready)
            u_max      = float(jax.device_get(u_max_jax))
            actual_cfl = u_max * dt / min_dx
            cfl_warn   = actual_cfl > max_cfl * 1.05

            # Probe sampling — batch transfer via device_get
            if probe_arr is not None:
                probe_data = extract_probe_data(state, probe_arr, grid)
                probe_host = jax.device_get(probe_data)
                _comp_names = {0: 'u', 1: 'v', 2: 'w'}
                for pi in range(len(probe_list)):
                    for cname_key, cname_out in _comp_names.items():
                        key = f'u_{cname_key}'
                        if key in probe_host:
                            mon_vals[f'probe_{pi}_{cname_out}'] = float(probe_host[key][pi])
                    if 'p' in probe_host:
                        mon_vals[f'probe_{pi}_p'] = float(probe_host['p'][pi])
                    if 'T' in probe_host:
                        mon_vals[f'probe_{pi}_T'] = float(probe_host['T'][pi])

            # Write force diagnostics to HDF5
            if do_forces and state.ibm_forces is not None:
                forces = np.asarray(jax.device_get(state.ibm_forces))
                _axes_n = ['x', 'y', 'z']
                diag_kw = {}
                # Raw IBM forces (informational — includes virtual-fluid inertia)
                for i in range(grid.ndim):
                    diag_kw[f'F{_axes_n[i]}'] = float(forces[i])

                # Pressure+viscous Cd/Cl (accurate aerodynamic coefficients)
                if pressure_cd_fn is not None:
                    diag_kw['Cd'] = _pcd_host.get('Cd', 0.0)
                    diag_kw['Cl'] = _pcd_host.get('Cl', 0.0)
                    if 'Cd_p' in _pcd_host:
                        diag_kw['Cd_p'] = _pcd_host['Cd_p']
                        diag_kw['Cd_v'] = _pcd_host['Cd_v']
                        diag_kw['Cl_p'] = _pcd_host['Cl_p']
                        diag_kw['Cl_v'] = _pcd_host['Cl_v']
                    if grid.ndim > 2:
                        diag_kw['Cz'] = _pcd_host.get('Cz', 0.0)
                else:
                    # Fallback: IBM direct-forcing (O(1/dt) — approximate only)
                    diag_kw['Cd'] = float(forces[0]) / q_ref
                    diag_kw['Cl'] = float(forces[1]) / q_ref if grid.ndim > 1 else 0.0
                    if grid.ndim > 2:
                        diag_kw['Cz'] = float(forces[2]) / q_ref
                writer.append_diagnostics(step, float(t_sim), **diag_kw)

            # Write monitor data to HDF5 every diag_every steps
            writer.append_monitor(step, float(t_sim), mon_vals)

            # Collect Cl history for Strouhal estimation
            if _st_enabled and 'Cl' in mon_vals:
                _cl_history_t.append(float(t_sim))
                _cl_history_v.append(mon_vals['Cl'])

            # Convergence history
            if _conv_enabled and _conv_qty in mon_vals:
                _conv_history.append(mon_vals[_conv_qty])

            # Invoke user callbacks (visualization, logging, etc.)
            for cb in callbacks:
                try:
                    cb(step=step, t=float(t_sim), state=state,
                       monitor=mon_vals, grid=grid, cfl=actual_cfl, dt=dt)
                except Exception:
                    pass  # never abort simulation due to callback error

            # Field snapshot + console row (every save_every steps)
            if step % save_every == 0:
                writer.write_snapshot(step, float(t_sim), state)
                elapsed = time.time() - t_wall_start
                pct     = 100.0 * (float(t_sim) - t_start) / (t_end - t_start)

                # Strouhal estimation from Cl FFT (need at least 32 samples)
                if _st_enabled and len(_cl_history_v) >= 32:
                    _cl_arr = np.array(_cl_history_v)
                    _dt_diag = np.mean(np.diff(_cl_history_t))
                    _freqs = np.fft.rfftfreq(len(_cl_arr), d=_dt_diag)
                    _power = np.abs(np.fft.rfft(_cl_arr - _cl_arr.mean())) ** 2
                    _ipeak = int(np.argmax(_power[1:])) + 1
                    _st_value = float(_freqs[_ipeak]) * _st_D / U_ref

                mon_str = ''.join(f'  {mon_vals.get(q, float("nan")):9.4f}'
                                  for q in monitor)
                cfl_tag = f'{actual_cfl:5.3f}' + ('!' if cfl_warn else ' ')
                _st_tag = f'  St={_st_value:.4f}' if _st_enabled and not np.isnan(_st_value) else ''
                _p(f"{step:8d}  {float(t_sim):8.3f}{mon_str}  "
                   f"{cfl_tag}  {pct:5.1f}  {elapsed:7.1f}{_st_tag}")

                # Early stopping check
                if _conv_enabled and len(_conv_history) >= _conv_window:
                    _recent = np.array(_conv_history[-_conv_window:])
                    _rel_std = float(np.std(_recent) / max(abs(np.mean(_recent)), 1e-12))
                    if _rel_std < _conv_tol:
                        _converged = True
                        _p(f"\n[Runner] Converged: "
                           f"{_conv_qty} relative std = {_rel_std:.2e} "
                           f"< tol = {_conv_tol:.2e} "
                           f"(window = {_conv_window} samples)")
                        break

        # Ensure a final snapshot exists for reliable restart
        if step % save_every != 0:
            writer.write_snapshot(step, float(t_sim), state)
            _p(f"[Runner] Final snapshot written at step {step}, t={float(t_sim):.3f}")

    # Final Strouhal report
    if _st_enabled and not np.isnan(_st_value):
        _p(f"[Runner] Final Strouhal estimate: St = {_st_value:.4f}")

    if _converged:
        _p(f"[Runner] Simulation stopped early due to convergence at t={float(t_sim):.3f}")
    _p(f"\n[Runner] Finished.  Output: {h5_path}\n")
    return state


# ── CLI entry point ────────────────────────────────────────────────────────

def _load_config_from_dir(case_dir: str) -> dict:
    """Dynamically import CONFIG from ``<case_dir>/__init__.py``.

    Uses ``submodule_search_locations`` so relative imports
    (``from .domain import DOMAIN``) work inside the package.
    """
    import sys
    case_path = Path(case_dir).resolve()
    init_file = case_path / '__init__.py'
    if not init_file.exists():
        raise FileNotFoundError(f"No __init__.py found in '{case_dir}'")
    spec = importlib.util.spec_from_file_location(
        '_case', str(init_file),
        submodule_search_locations=[str(case_path)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_case'] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop('_case', None)
    if not hasattr(mod, 'CONFIG'):
        raise AttributeError(f"'{init_file}' must define a 'CONFIG' dict")
    return mod.CONFIG


def main():
    """CLI entry point: ``diffflow <case_dir> [--restart] [--dry-run]``."""
    import argparse
    parser = argparse.ArgumentParser(description='DiffFlow simulation runner')
    parser.add_argument('case_dir', help='Path to case directory (must contain __init__.py)')
    parser.add_argument('--restart', action='store_true',
                        help='Restart from the existing output file in the case directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print simulation parameters and resource estimates without running')
    args = parser.parse_args()

    _cfg = _load_config_from_dir(args.case_dir)

    _restart_path = None
    if args.restart:
        out_cfg = _cfg.get('output', {})
        _restart_path = str(
            Path(out_cfg.get('dir', 'output')) / out_cfg.get('file', 'run.h5')
        )

    run(_cfg, restart_from=_restart_path, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
