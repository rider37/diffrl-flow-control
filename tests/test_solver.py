"""Tests for the fractional-step solver."""
import jax.numpy as jnp
import pytest

from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.state import FluidState
from jax_cfd.core.solver import create_step_fn
from jax_cfd.core.operators import divergence
from jax_cfd.core.boundaries import apply_boundaries, pad_velocity_for_diffusion
from jax_cfd.core.turbulence import get_turbulence_model
from jax_cfd.core.schemes import get_pressure_solver, make_advection_scheme


def _make_solver_and_state(grid, nu=0.01, time_scheme='rk3'):
    """Build a simple solver with wall BCs and return (step_fn, state)."""
    ndim = grid.ndim
    bc_cfg = {}
    for i, ax in enumerate(['x', 'y', 'z'][:ndim]):
        bc_cfg[f'{ax}_min'] = {'type': 'wall_no_slip'}
        bc_cfg[f'{ax}_max'] = {'type': 'wall_no_slip'}

    def apply_bc(vel, _cfg, g, inlet_vel=None):
        return apply_boundaries(vel, bc_cfg, g)

    def pad_diff(vel, g, inlet_vel=None):
        return pad_velocity_for_diffusion(vel, bc_cfg, g)

    turb_fn = get_turbulence_model('laminar')
    advect_fn = make_advection_scheme('muscl')
    solve_p = get_pressure_solver('fft')

    step_fn = create_step_fn(
        grid=grid,
        nu_base=nu,
        apply_bc_fn=apply_bc,
        pad_diff_fn=pad_diff,
        get_nu_t_fn=turb_fn,
        advect_fn=advect_fn,
        solve_p_fn=solve_p,
        time_scheme=time_scheme,
    )

    vel = tuple(jnp.zeros(grid.face_shape(i)) for i in range(ndim))
    vel = apply_bc(vel, {}, grid)
    p = jnp.zeros(grid.cell_shape())
    state = FluidState(velocity=vel, pressure=p, scalars={}, ibm_forces=None)

    return step_fn, state


class TestSolverBasic2D:
    def test_step_returns_fluid_state(self, grid_2d):
        step_fn, state = _make_solver_and_state(grid_2d)
        result = step_fn(state, 0.001, 0.0)
        assert isinstance(result, FluidState)
        assert len(result.velocity) == 2

    def test_zero_flow_stays_zero(self, grid_2d):
        """Zero velocity + no forcing should remain zero."""
        step_fn, state = _make_solver_and_state(grid_2d)
        for _ in range(5):
            state = step_fn(state, 0.001, 0.0)
        for u_comp in state.velocity:
            assert jnp.allclose(u_comp, 0.0, atol=1e-8)

    def test_divergence_free(self, grid_2d):
        """After stepping, velocity should be approximately divergence-free."""
        step_fn, state = _make_solver_and_state(grid_2d)
        # Give it a non-trivial initial velocity
        u = jnp.ones(grid_2d.face_shape(0)) * 0.5
        v = jnp.zeros(grid_2d.face_shape(1))
        state = FluidState(velocity=(u, v), pressure=state.pressure,
                           scalars={}, ibm_forces=None)
        state = step_fn(state, 0.001, 0.0)
        div = divergence(state.velocity, grid_2d)
        assert float(jnp.max(jnp.abs(div))) < 1e-4

    def test_euler_scheme(self, grid_2d):
        step_fn, state = _make_solver_and_state(grid_2d, time_scheme='euler')
        result = step_fn(state, 0.001, 0.0)
        assert isinstance(result, FluidState)


class TestSolverBasic3D:
    def test_3d_step(self, grid_3d):
        step_fn, state = _make_solver_and_state(grid_3d)
        result = step_fn(state, 0.001, 0.0)
        assert len(result.velocity) == 3

    def test_3d_divergence_free(self, grid_3d):
        step_fn, state = _make_solver_and_state(grid_3d)
        state = step_fn(state, 0.001, 0.0)
        div = divergence(state.velocity, grid_3d)
        assert float(jnp.max(jnp.abs(div))) < 1e-4
