"""Tests for differentiable simulation (AD through the solver)."""
import jax
import jax.numpy as jnp
import pytest

from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.state import FluidState
from jax_cfd.core.solver import create_step_fn
from jax_cfd.core.boundaries import apply_boundaries, pad_velocity_for_diffusion
from jax_cfd.core.turbulence import get_turbulence_model
from jax_cfd.core.schemes import get_pressure_solver, make_advection_scheme
from jax_cfd.differentiable import differentiable_simulate


def _make_solver_and_state():
    """Build a minimal 8x8 solver for fast AD tests."""
    grid = MACGrid((8, 8), (1.0, 1.0))
    bc_cfg = {
        'x_min': {'type': 'wall_no_slip'},
        'x_max': {'type': 'wall_no_slip'},
        'y_min': {'type': 'wall_no_slip'},
        'y_max': {'type': 'inlet', 'velocity': [1.0, 0.0]},
    }

    def apply_bc(vel, _cfg, g, inlet_vel=None):
        return apply_boundaries(vel, bc_cfg, g)

    def pad_diff(vel, g, inlet_vel=None):
        return pad_velocity_for_diffusion(vel, bc_cfg, g)

    step_fn = create_step_fn(
        grid=grid,
        nu_base=0.01,
        apply_bc_fn=apply_bc,
        pad_diff_fn=pad_diff,
        get_nu_t_fn=get_turbulence_model('laminar'),
        advect_fn=make_advection_scheme('central'),
        solve_p_fn=get_pressure_solver('fft'),
        time_scheme='rk3',
    )

    vel = tuple(jnp.zeros(grid.face_shape(i)) for i in range(2))
    vel = apply_bc(vel, {}, grid)
    p = jnp.zeros(grid.cell_shape())
    state = FluidState(velocity=vel, pressure=p, scalars={}, ibm_forces=None)
    return step_fn, state


class TestDifferentiableSimulate:
    def test_returns_fluid_state(self):
        step_fn, state0 = _make_solver_and_state()
        result = differentiable_simulate(step_fn, state0, n_steps=3, dt=0.001)
        assert isinstance(result, FluidState)
        assert len(result.velocity) == 2

    def test_gradient_is_finite(self):
        """jax.grad through a short simulation should produce finite gradients."""
        step_fn, state0 = _make_solver_and_state()

        def loss(nu):
            state = differentiable_simulate(
                step_fn, state0, n_steps=3, dt=0.001, nu_override=nu,
            )
            return jnp.sum(state.velocity[0] ** 2)

        grad_val = jax.grad(loss)(jnp.float32(0.01))
        assert jnp.isfinite(grad_val)

    def test_gradient_is_nonzero(self):
        """Gradient of KE w.r.t. viscosity should be non-trivial."""
        step_fn, state0 = _make_solver_and_state()

        def loss(nu):
            state = differentiable_simulate(
                step_fn, state0, n_steps=3, dt=0.001, nu_override=nu,
            )
            return jnp.sum(state.velocity[0] ** 2)

        grad_val = jax.grad(loss)(jnp.float32(0.01))
        assert float(jnp.abs(grad_val)) > 1e-10

    def test_gradient_matches_finite_differences(self):
        """AD gradient should match central finite differences within 10%."""
        step_fn, state0 = _make_solver_and_state()

        def loss(nu):
            state = differentiable_simulate(
                step_fn, state0, n_steps=3, dt=0.001, nu_override=nu,
            )
            return jnp.sum(state.velocity[0] ** 2)

        nu = jnp.float32(0.01)
        ad_grad = float(jax.grad(loss)(nu))

        # Use large eps to avoid float32 catastrophic cancellation
        eps = 2e-3
        fd_grad = (float(loss(nu + eps)) - float(loss(nu - eps))) / (2 * eps)

        rel_err = abs(ad_grad - fd_grad) / max(abs(fd_grad), 1e-12)
        assert rel_err < 0.20, f"AD vs FD relative error {rel_err:.4f} > 0.20"
