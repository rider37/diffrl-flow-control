import jax.numpy as jnp
from typing import Dict, Callable, Tuple, Optional
from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.state import FluidState
from jax_cfd.core.operators import (
    divergence, gradient_p as _default_gradient_p, diff_axis_padded_nu,
    strain_rate_magnitude,
)
from jax_cfd.core.advection import advect_scalar

# ======================================================================
# Advanced Fractional Step Navier-Stokes Solver (MAC Grid)
# ======================================================================

def create_step_fn(
    grid: MACGrid,
    nu_base: float,
    apply_bc_fn: Callable,
    pad_diff_fn: Callable,
    get_nu_t_fn: Callable,
    advect_fn: Callable,
    solve_p_fn: Callable,
    ibm_fn: Callable = None,
    time_scheme: str = 'rk3',
    solve_scalars: bool = False,
    solve_buoyancy: bool = False,
    gravity_vector: Tuple[float, ...] = (0.0, -9.81),
    beta: float = 0.003,
    T_ref: float = 0.0,
    advect_scalar_fn: Callable = None,
    pad_scalar_fn: Callable = None,
    scalar_params: Dict[str, Dict] = None,
    body_force: Optional[Tuple[float, ...]] = None,
    gradient_p_fn: Optional[Callable] = None,
    ibm_scalar_fn: Callable = None,
    scalar_source: Dict[str, jnp.ndarray] = None,
    implicit_diffusion_fn: Callable = None,
    scalar_rhs_fn: Callable = None,
    momentum_source_fn: Callable = None,
    inlet_schedule_fn: Callable = None,
    rheology_fn: Callable = None,
):
    """
    Fractional-Step incompressible NS solver on a staggered MAC grid.

    RK3 buoyancy coupling
    ---------------------
    Each RK3 stage receives the temperature field *at that stage* (s_1, s_2),
    not the initial time-n field.  This elevates the T-u coupling from 1st-order
    (explicit Euler) to 3rd-order (consistent with the Shu-Osher TVD-RK3 scheme).

    body_force : optional constant acceleration (fx, fy, ...) added to every RHS
        evaluation.  Use for pressure-gradient-driven flows such as channel flow.

    gradient_p_fn : optional replacement for the default ``gradient_p`` operator.
        Pass a partial with ``periodic_axes`` for domains with periodic BCs so that
        the pressure gradient is computed with wrap-around padding.

    implicit_diffusion_fn : optional callable ``(vel, dt_impl) -> vel``.
        When provided, Crank-Nicolson splitting is used: the explicit diffusion
        uses ``0.5*nu_base + nu_t``, and the implicit half is solved by this
        function with ``dt_impl = 0.5 * dt_substep``.

    scalar_rhs_fn : optional callable
        ``(scalars, vel, state, nu_base) -> Dict[str, array]``.
        When provided, replaces ``compute_scalar_rhs`` for the scalar update.
        Used by the Cahn-Hilliard phase-field solver.

    momentum_source_fn : optional callable
        ``(scalars, velocity, grid) -> tuple of face-centred arrays``.
        Extra body force (e.g. surface tension) added to momentum RHS.

    inlet_schedule_fn : optional callable ``(t) -> tuple of floats``.
        Time-varying inlet velocity schedule.  When provided, the inlet BC
        velocity is overridden at each time step using this function.
    """
    _grad_p = gradient_p_fn if gradient_p_fn is not None else _default_gradient_p
    _use_implicit = implicit_diffusion_fn is not None
    # Explicit fraction of nu_base: full if explicit-only, half if CN
    _nu_base_explicit = 0.5 * nu_base if _use_implicit else nu_base
    # Scalar RHS function: custom (CH / boiling) or default advection-diffusion
    _scalar_rhs = scalar_rhs_fn if scalar_rhs_fn is not None else None
    _use_rheology = rheology_fn is not None

    # -- Semi-implicit Brinkman helpers ----------------------------------------
    def _interpolate_brinkman_to_faces(brinkman_field):
        """Cell-centred alpha -> face-centred, one per velocity component."""
        alpha_faces = []
        for i in range(grid.ndim):
            pad_b = [(0, 0)] * grid.ndim
            pad_b[i] = (1, 1)
            alpha_padded = jnp.pad(brinkman_field, pad_b, mode='edge')
            sl_p = [slice(None)] * grid.ndim
            sl_m = [slice(None)] * grid.ndim
            sl_p[i] = slice(1, None)
            sl_m[i] = slice(None, -1)
            alpha_faces.append(
                0.5 * (alpha_padded[tuple(sl_p)] + alpha_padded[tuple(sl_m)]))
        return tuple(alpha_faces)

    def _apply_brinkman_implicit(vel_star, brinkman_field, dt_coeff):
        """Semi-implicit Brinkman: u* = u* / (1 + dt_coeff * alpha_face).

        Unconditionally stable for any alpha_max.
        """
        if brinkman_field is None:
            return vel_star
        alpha_faces = _interpolate_brinkman_to_faces(brinkman_field)
        return tuple(
            u / (1.0 + dt_coeff * a)
            for u, a in zip(vel_star, alpha_faces))

    def compute_N_S_rhs(
        velocity: Tuple[jnp.ndarray, ...],
        scalars: Dict[str, jnp.ndarray],
        state: FluidState,
        nu_base_val=_nu_base_explicit,
        inlet_vel=None,
    ) -> Tuple[jnp.ndarray, ...]:
        """Computes R(u) = -(u·∇)u + ν_eff ∇²u + buoyancy(scalars)"""
        nu_t = get_nu_t_fn(state, grid, {})
        if _use_rheology:
            strain_mag = strain_rate_magnitude(velocity, grid)
            nu_eff = rheology_fn(strain_mag) + nu_t
        else:
            nu_eff = nu_base_val + nu_t

        # Advection (using injected scheme: central or MUSCL)
        adv = advect_fn(velocity, grid)

        # Pre-compute momentum source (e.g. surface tension) once for all axes
        _mom_src = momentum_source_fn(scalars, velocity, grid) if momentum_source_fn is not None else None

        # Diffusion with ghost-cell padding
        vel_padded = pad_diff_fn(velocity, grid, inlet_vel=inlet_vel)
        rhs_list = []
        for i, (u_cmp, u_pad) in enumerate(zip(velocity, vel_padded)):

            # Interpolate nu_eff (cell-centred) to i-faces
            pad_i = [(0, 0)] * grid.ndim
            pad_i[i] = (1, 1)
            nu_eff_padded = jnp.pad(nu_eff, pad_i, mode='edge')
            slices_p = [slice(None)] * grid.ndim
            slices_m = [slice(None)] * grid.ndim
            slices_p[i] = slice(1, None)
            slices_m[i] = slice(None, -1)
            nu_eff_face = 0.5 * (nu_eff_padded[tuple(slices_p)] + nu_eff_padded[tuple(slices_m)])

            diff = 0.0
            for j in range(grid.ndim):
                slices_extract = [slice(None)] * grid.ndim
                for k in range(grid.ndim):
                    if k != j:
                        slices_extract[k] = slice(1, -1)
                u_pad_j = u_pad[tuple(slices_extract)]
                diff += diff_axis_padded_nu(u_pad_j, j, grid) * nu_eff_face

            total_rhs = -adv[i] + diff

            # Buoyancy: use the *current stage* scalars, not the time-n state
            if solve_buoyancy and 'T' in scalars:
                g_i = gravity_vector[i] if i < len(gravity_vector) else 0.0
                if abs(g_i) > 1e-10:
                    pad_w = [(0, 0)] * grid.ndim
                    pad_w[i] = (1, 1)
                    T_pad = jnp.pad(scalars['T'], pad_w, mode='edge')
                    slices_p2 = [slice(None)] * grid.ndim
                    slices_p2[i] = slice(1, None)
                    slices_m2 = [slice(None)] * grid.ndim
                    slices_m2[i] = slice(None, -1)
                    T_face = 0.5 * (T_pad[tuple(slices_p2)] + T_pad[tuple(slices_m2)])
                    total_rhs += g_i * beta * (T_face - T_ref)

            # Constant body force (e.g. channel-flow pressure gradient)
            if body_force is not None and i < len(body_force):
                total_rhs = total_rhs + body_force[i]

            # Momentum source (e.g. surface tension from phase-field)
            if _mom_src is not None:
                total_rhs = total_rhs + _mom_src[i]

            # Brinkman penalization is handled semi-implicitly after the
            # predictor step — see _apply_brinkman_implicit().

            rhs_list.append(total_rhs)

        return tuple(rhs_list)

    def compute_scalar_rhs(
        scalars_curr: Dict[str, jnp.ndarray],
        vel_curr: Tuple[jnp.ndarray, ...],
        state: FluidState,
        nu_base_val=nu_base,
        kappa_field=None,
        extra=None,
    ) -> Dict[str, jnp.ndarray]:
        """Explicit RHS (advection + diffusion) for all scalars."""
        nu_t = get_nu_t_fn(state, grid, {})
        s_rhs = {}
        for name, s_n in scalars_curr.items():
            # Advection
            if advect_scalar_fn is not None:
                adv_s = advect_scalar_fn(s_n, vel_curr, grid)
            else:
                adv_s = advect_scalar(s_n, vel_curr, grid)

            # Effective diffusivity kappa = nu_lam/Pr + nu_t/Pr_t
            params = (scalar_params or {}).get(name, {})
            Pr = float(params.get('Pr', params.get('Le', 1.0)))
            Pr_t = 0.85
            kappa = nu_base_val / Pr + nu_t / Pr_t
            if kappa_field is not None:
                kappa = kappa + kappa_field

            # Diffusion with ghost-cell BC
            diff_s = 0.0
            s_all_padded = pad_scalar_fn(s_n, name, grid, extra) if pad_scalar_fn is not None else None
            for j in range(grid.ndim):
                if s_all_padded is not None:
                    slices_extract = [slice(None)] * grid.ndim
                    for k in range(grid.ndim):
                        if k != j:
                            slices_extract[k] = slice(1, -1)
                    s_pad_j = s_all_padded[tuple(slices_extract)]
                else:
                    pad_width = [(0, 0)] * grid.ndim
                    pad_width[j] = (1, 1)
                    s_pad_j = jnp.pad(s_n, pad_width, mode='edge')
                diff_s += diff_axis_padded_nu(s_pad_j, j, grid) * kappa

            s_rhs[name] = -adv_s + diff_s
            if scalar_source is not None and name in scalar_source:
                s_rhs[name] = s_rhs[name] + scalar_source[name]
        return s_rhs

    def fractional_substep(
        vel_curr: Tuple[jnp.ndarray, ...],
        scalars_curr: Dict[str, jnp.ndarray],
        state: FluidState,
        dt_sub: float,
        extra,
        nu_base_val=_nu_base_explicit,
        brinkman_field=None,
        inlet_vel=None,
        kappa_field=None,
    ) -> Tuple[Tuple[jnp.ndarray, ...], jnp.ndarray, Dict[str, jnp.ndarray], Optional[jnp.ndarray]]:
        """Predictor -> Poisson -> corrector, plus explicit scalar update."""
        # Pass scalars_curr so buoyancy uses the current-stage temperature
        rhs_v = compute_N_S_rhs(vel_curr, scalars_curr, state, nu_base_val, inlet_vel=inlet_vel)

        if solve_scalars:
            if _scalar_rhs is not None:
                rhs_s = _scalar_rhs(scalars_curr, vel_curr, state, nu_base_val)
            else:
                rhs_s = compute_scalar_rhs(scalars_curr, vel_curr, state, nu_base_val, kappa_field=kappa_field, extra=extra)

        # Predictor
        vel_star = tuple(u + dt_sub * r for u, r in zip(vel_curr, rhs_v))
        # Semi-implicit Brinkman: u* = u* / (1 + dt*alpha)
        vel_star = _apply_brinkman_implicit(vel_star, brinkman_field, dt_sub)
        # CN implicit diffusion half-step (before BCs so BCs are re-applied after)
        if _use_implicit:
            vel_star = implicit_diffusion_fn(vel_star, 0.5 * dt_sub)
        vel_star = apply_bc_fn(vel_star, {}, grid, inlet_vel=inlet_vel)

        # Scalar update (forward Euler within the substep)
        if solve_scalars:
            scalars_next = {name: s + dt_sub * rhs_s[name] for name, s in scalars_curr.items()}
            if ibm_scalar_fn is not None:
                scalars_next = ibm_scalar_fn(scalars_next)
        else:
            scalars_next = scalars_curr

        # Optional IBM
        ibm_forces = None
        if ibm_fn is not None:
            vel_star, ibm_forces = ibm_fn(vel_star, extra, grid, dt_sub)

        # Pressure projection
        div_star = divergence(vel_star, grid)
        p_next = solve_p_fn(div_star / dt_sub, grid)

        # Corrector
        grad_p = _grad_p(p_next, grid)
        vel_next = tuple(u_star - dt_sub * dp for u_star, dp in zip(vel_star, grad_p))
        vel_next = apply_bc_fn(vel_next, {}, grid, inlet_vel=inlet_vel)

        return vel_next, p_next, scalars_next, ibm_forces

    # -- Main time-stepping function ------------------------------------------

    def step_fn(state: FluidState, dt: float, extra,
                nu_override=None, brinkman_field=None,
                kappa_field=None) -> FluidState:
        """Advance state by one time step.

        Parameters
        ----------
        nu_override : float or jnp scalar, optional
            When provided, replaces the closure-captured ``nu_base`` for
            this step.  This makes the step function differentiable w.r.t.
            viscosity via ``jax.grad``.

            .. note:: ``nu_override`` is **not compatible** with implicit
               diffusion (Crank-Nicolson), which uses a pre-built solver
               with the original ``nu_base``.

        brinkman_field : jnp.ndarray, optional
            Cell-centred Brinkman penalization field alpha(x).  When provided,
            adds ``-alpha*u`` to the momentum RHS (topology optimisation).
            Differentiable via ``jax.grad``.

        kappa_field : jnp.ndarray, optional
            Cell-centred extra thermal diffusivity field.  When provided,
            added to the base scalar diffusivity (kappa_base + kappa_field).
            Use for spatially-varying conductivity (e.g. solid regions in
            conjugate heat transfer topology optimisation).
        """
        # Resolve effective explicit viscosity
        if nu_override is not None:
            _nu_val = 0.5 * nu_override if _use_implicit else nu_override
        else:
            _nu_val = _nu_base_explicit

        # Compute inlet velocity from schedule (if provided)
        _inlet_vel = inlet_schedule_fn(extra) if inlet_schedule_fn is not None else None

        v_n = state.velocity
        s_n = state.scalars

        if time_scheme.lower() == 'rk3':
            # Shu-Osher TVD-RK3:
            #   Stage 1: u(1) = u^n + dt*R(u^n, T^n)
            #   Stage 2: u(2) = 3/4 u^n + 1/4 u(1) + 1/4 dt*R(u(1), T(1))
            #   Stage 3: u^n+1 = 1/3 u^n + 2/3 u(2) + 2/3 dt*R(u(2), T(2))
            # T is advanced with the same blending coefficients so the
            # buoyancy term is always evaluated at the matching stage.

            # Stage 1
            v_1, p_1, s_1, f_1 = fractional_substep(v_n, s_n, state, dt, extra, _nu_val, brinkman_field, inlet_vel=_inlet_vel, kappa_field=kappa_field)

            # Stage 2 -- uses s_1 for buoyancy
            rhs_1_v = compute_N_S_rhs(v_1, s_1, state, _nu_val, inlet_vel=_inlet_vel)
            if solve_scalars:
                if _scalar_rhs is not None:
                    rhs_1_s = _scalar_rhs(s_1, v_1, state, _nu_val)
                else:
                    rhs_1_s = compute_scalar_rhs(s_1, v_1, state, _nu_val, kappa_field=kappa_field, extra=extra)

            v_2_star = tuple(
                0.75 * u_n + 0.25 * u_1 + 0.25 * dt * r1
                for u_n, u_1, r1 in zip(v_n, v_1, rhs_1_v)
            )
            v_2_star = _apply_brinkman_implicit(v_2_star, brinkman_field, 0.25 * dt)
            if _use_implicit:
                v_2_star = implicit_diffusion_fn(v_2_star, 0.5 * 0.25 * dt)
            v_2_star = apply_bc_fn(v_2_star, {}, grid, inlet_vel=_inlet_vel)

            if ibm_fn is not None:
                v_2_star, _ = ibm_fn(v_2_star, extra, grid, 0.25 * dt)

            p_2 = solve_p_fn(divergence(v_2_star, grid) / (0.25 * dt), grid)
            v_2 = tuple(
                u_s - 0.25 * dt * dp
                for u_s, dp in zip(v_2_star, _grad_p(p_2, grid))
            )
            v_2 = apply_bc_fn(v_2, {}, grid, inlet_vel=_inlet_vel)

            if solve_scalars:
                s_2 = {
                    name: 0.75 * s_n[name] + 0.25 * s_1[name] + 0.25 * dt * rhs_1_s[name]
                    for name in s_n
                }
                if ibm_scalar_fn is not None:
                    s_2 = ibm_scalar_fn(s_2)
            else:
                s_2 = s_n

            # Stage 3 -- uses s_2 for buoyancy
            rhs_2_v = compute_N_S_rhs(v_2, s_2, state, _nu_val, inlet_vel=_inlet_vel)
            if solve_scalars:
                if _scalar_rhs is not None:
                    rhs_2_s = _scalar_rhs(s_2, v_2, state, _nu_val)
                else:
                    rhs_2_s = compute_scalar_rhs(s_2, v_2, state, _nu_val, kappa_field=kappa_field, extra=extra)

            v_3_star = tuple(
                (1.0 / 3.0) * u_n + (2.0 / 3.0) * u_2 + (2.0 / 3.0) * dt * r2
                for u_n, u_2, r2 in zip(v_n, v_2, rhs_2_v)
            )
            v_3_star = _apply_brinkman_implicit(v_3_star, brinkman_field, (2.0 / 3.0) * dt)
            if _use_implicit:
                v_3_star = implicit_diffusion_fn(v_3_star, 0.5 * (2.0 / 3.0) * dt)
            v_3_star = apply_bc_fn(v_3_star, {}, grid, inlet_vel=_inlet_vel)

            f_next = None
            if ibm_fn is not None:
                v_3_star, f_next = ibm_fn(v_3_star, extra, grid, (2.0 / 3.0) * dt)

            p_next = solve_p_fn(
                divergence(v_3_star, grid) / ((2.0 / 3.0) * dt), grid
            )
            v_next = tuple(
                u_s - (2.0 / 3.0) * dt * dp
                for u_s, dp in zip(v_3_star, _grad_p(p_next, grid))
            )
            v_next = apply_bc_fn(v_next, {}, grid, inlet_vel=_inlet_vel)

            if solve_scalars:
                scalars_next = {
                    name: (1.0 / 3.0) * s_n[name] + (2.0 / 3.0) * s_2[name]
                           + (2.0 / 3.0) * dt * rhs_2_s[name]
                    for name in s_n
                }
                if ibm_scalar_fn is not None:
                    scalars_next = ibm_scalar_fn(scalars_next)
            else:
                scalars_next = s_n

            final_forces = f_next

        else:  # Explicit Euler
            v_next, p_next, scalars_next, final_forces = fractional_substep(
                v_n, s_n, state, dt, extra, _nu_val, brinkman_field, inlet_vel=_inlet_vel, kappa_field=kappa_field
            )

        return FluidState(
            velocity=v_next,
            pressure=p_next,
            scalars=scalars_next,
            ibm_forces=final_forces,
        )

    return step_fn
