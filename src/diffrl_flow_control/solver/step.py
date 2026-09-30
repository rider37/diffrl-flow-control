"""Fractional-step time integration of the incompressible Navier--Stokes
equations (optionally Boussinesq) on a staggered MAC grid.

Each step uses the three-stage Shu--Osher TVD Runge--Kutta scheme.  Every stage
is a predictor (advection + diffusion + buoyancy), an optional immersed-boundary
correction, a pressure projection and a velocity correction.  The temperature is
advanced with the same stage coefficients, so the buoyancy term always uses the
temperature of the current stage.
"""
from typing import Callable, Dict, Tuple

import jax.numpy as jnp

from diffrl_flow_control.solver.advection import advect_scalar
from diffrl_flow_control.solver.grid import MACGrid
from diffrl_flow_control.solver.operators import diff_axis_padded_nu, divergence, gradient_p
from diffrl_flow_control.solver.state import FluidState


def create_step_fn(
    grid: MACGrid,
    nu_base: float,
    apply_bc_fn: Callable,
    pad_diff_fn: Callable,
    get_nu_t_fn: Callable,
    advect_fn: Callable,
    solve_p_fn: Callable,
    ibm_fn: Callable = None,
    solve_scalars: bool = False,
    solve_buoyancy: bool = False,
    gravity_vector: Tuple[float, ...] = (0.0, -9.81),
    beta: float = 0.003,
    T_ref: float = 0.0,
    advect_scalar_fn: Callable = None,
    pad_scalar_fn: Callable = None,
    scalar_params: Dict[str, Dict] = None,
):
    """Return ``step_fn(state, dt, extra) -> state`` advancing one time step.

    ``extra`` is passed to ``ibm_fn`` (the jet actuation of the cylinder) and to
    ``pad_scalar_fn`` (the heater temperatures of the Rayleigh--Benard wall).
    """

    def compute_N_S_rhs(velocity, scalars, state, nu_base_val=nu_base):
        """R(u) = -(u.grad)u + nu lap(u) + buoyancy(T)."""
        nu_t = get_nu_t_fn(state, grid, {})
        nu_eff = nu_base_val + nu_t

        adv = advect_fn(velocity, grid)

        vel_padded = pad_diff_fn(velocity, grid)
        rhs_list = []
        for i, (u_cmp, u_pad) in enumerate(zip(velocity, vel_padded)):

            # viscosity interpolated to the i-faces
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

            # buoyancy from the temperature of the current stage
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

            rhs_list.append(total_rhs)

        return tuple(rhs_list)

    def compute_scalar_rhs(scalars_curr, vel_curr, state, nu_base_val=nu_base, extra=None):
        """Explicit advection--diffusion right-hand side of every scalar."""
        nu_t = get_nu_t_fn(state, grid, {})
        s_rhs = {}
        for name, s_n in scalars_curr.items():
            if advect_scalar_fn is not None:
                adv_s = advect_scalar_fn(s_n, vel_curr, grid)
            else:
                adv_s = advect_scalar(s_n, vel_curr, grid)

            # diffusivity kappa = nu / Pr
            params = (scalar_params or {}).get(name, {})
            Pr = float(params.get('Pr', params.get('Le', 1.0)))
            Pr_t = 0.85
            kappa = nu_base_val / Pr + nu_t / Pr_t

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
        return s_rhs

    def fractional_substep(vel_curr, scalars_curr, state, dt_sub, extra, nu_base_val=nu_base):
        """First RK3 stage: predictor -> immersed boundary -> projection -> corrector."""
        rhs_v = compute_N_S_rhs(vel_curr, scalars_curr, state, nu_base_val)
        if solve_scalars:
            rhs_s = compute_scalar_rhs(scalars_curr, vel_curr, state, nu_base_val, extra=extra)

        vel_star = tuple(u + dt_sub * r for u, r in zip(vel_curr, rhs_v))
        vel_star = apply_bc_fn(vel_star, {}, grid)

        if solve_scalars:
            scalars_next = {name: s + dt_sub * rhs_s[name] for name, s in scalars_curr.items()}
        else:
            scalars_next = scalars_curr

        ibm_forces = None
        if ibm_fn is not None:
            vel_star, ibm_forces = ibm_fn(vel_star, extra, grid, dt_sub)

        div_star = divergence(vel_star, grid)
        p_next = solve_p_fn(div_star / dt_sub, grid)

        grad_p = gradient_p(p_next, grid)
        vel_next = tuple(u_star - dt_sub * dp for u_star, dp in zip(vel_star, grad_p))
        vel_next = apply_bc_fn(vel_next, {}, grid)

        return vel_next, p_next, scalars_next, ibm_forces

    def step_fn(state: FluidState, dt: float, extra) -> FluidState:
        """Advance ``state`` by one time step of size ``dt``.

        Shu--Osher TVD-RK3:
          u(1)    = u^n + dt R(u^n, T^n)
          u(2)    = 3/4 u^n + 1/4 u(1) + 1/4 dt R(u(1), T(1))
          u^{n+1} = 1/3 u^n + 2/3 u(2) + 2/3 dt R(u(2), T(2)),
        each stage followed by the immersed-boundary correction and the projection.
        """
        _nu_val = nu_base
        v_n = state.velocity
        s_n = state.scalars

        # stage 1
        v_1, p_1, s_1, f_1 = fractional_substep(v_n, s_n, state, dt, extra, _nu_val)

        # stage 2
        rhs_1_v = compute_N_S_rhs(v_1, s_1, state, _nu_val)
        if solve_scalars:
            rhs_1_s = compute_scalar_rhs(s_1, v_1, state, _nu_val, extra=extra)

        v_2_star = tuple(
            0.75 * u_n + 0.25 * u_1 + 0.25 * dt * r1
            for u_n, u_1, r1 in zip(v_n, v_1, rhs_1_v)
        )
        v_2_star = apply_bc_fn(v_2_star, {}, grid)

        if ibm_fn is not None:
            v_2_star, _ = ibm_fn(v_2_star, extra, grid, 0.25 * dt)

        p_2 = solve_p_fn(divergence(v_2_star, grid) / (0.25 * dt), grid)
        v_2 = tuple(
            u_s - 0.25 * dt * dp
            for u_s, dp in zip(v_2_star, gradient_p(p_2, grid))
        )
        v_2 = apply_bc_fn(v_2, {}, grid)

        if solve_scalars:
            s_2 = {
                name: 0.75 * s_n[name] + 0.25 * s_1[name] + 0.25 * dt * rhs_1_s[name]
                for name in s_n
            }
        else:
            s_2 = s_n

        # stage 3
        rhs_2_v = compute_N_S_rhs(v_2, s_2, state, _nu_val)
        if solve_scalars:
            rhs_2_s = compute_scalar_rhs(s_2, v_2, state, _nu_val, extra=extra)

        v_3_star = tuple(
            (1.0 / 3.0) * u_n + (2.0 / 3.0) * u_2 + (2.0 / 3.0) * dt * r2
            for u_n, u_2, r2 in zip(v_n, v_2, rhs_2_v)
        )
        v_3_star = apply_bc_fn(v_3_star, {}, grid)

        f_next = None
        if ibm_fn is not None:
            v_3_star, f_next = ibm_fn(v_3_star, extra, grid, (2.0 / 3.0) * dt)

        p_next = solve_p_fn(
            divergence(v_3_star, grid) / ((2.0 / 3.0) * dt), grid
        )
        v_next = tuple(
            u_s - (2.0 / 3.0) * dt * dp
            for u_s, dp in zip(v_3_star, gradient_p(p_next, grid))
        )
        v_next = apply_bc_fn(v_next, {}, grid)

        if solve_scalars:
            scalars_next = {
                name: (1.0 / 3.0) * s_n[name] + (2.0 / 3.0) * s_2[name]
                       + (2.0 / 3.0) * dt * rhs_2_s[name]
                for name in s_n
            }
        else:
            scalars_next = s_n

        return FluidState(
            velocity=v_next,
            pressure=p_next,
            scalars=scalars_next,
            ibm_forces=f_next,
        )

    return step_fn
