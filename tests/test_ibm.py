"""Tests for the Immersed Boundary Method module."""
import jax.numpy as jnp
import pytest

from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.ibm import (
    peskin_phi,
    compute_tensor_weights_mac,
    interpolate_mac_velocity,
    spread_mac_forcing,
    apply_ibm_forcing,
)


@pytest.fixture
def grid():
    return MACGrid((16, 16), (1.0, 1.0))


@pytest.fixture
def circle_coords():
    """20 marker points on a circle at (0.5, 0.5), radius 0.2."""
    n = 20
    theta = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
    xs = 0.5 + 0.2 * jnp.cos(theta)
    ys = 0.5 + 0.2 * jnp.sin(theta)
    return (xs, ys)


class TestPeskinKernel:
    def test_non_negative(self):
        r = jnp.linspace(-3, 3, 200)
        assert jnp.all(peskin_phi(r) >= 0.0)

    def test_symmetric(self):
        r = jnp.linspace(0, 3, 100)
        assert jnp.allclose(peskin_phi(r), peskin_phi(-r), atol=1e-12)

    def test_compact_support(self):
        r = jnp.array([2.5, 3.0, 10.0])
        assert jnp.allclose(peskin_phi(r), 0.0)


class TestWeightsAndInterpolation:
    def test_weights_shape(self, grid, circle_coords):
        weights = compute_tensor_weights_mac(grid, circle_coords)
        n_markers = 20
        # 2 velocity components, each with 2 axis weight arrays
        assert len(weights) == 2
        for cmp in range(2):
            assert len(weights[cmp]) == 2
            # axis matching component: face count (N+1), other axis: cell count (N)
            assert weights[cmp][cmp].shape == (n_markers, 17)
            other = 1 - cmp
            assert weights[cmp][other].shape == (n_markers, 16)

    def test_interpolate_uniform_velocity(self, grid, circle_coords):
        """Uniform u=1 field should interpolate to ~1 at every marker."""
        u = jnp.ones(grid.face_shape(0))
        v = jnp.zeros(grid.face_shape(1))
        weights = compute_tensor_weights_mac(grid, circle_coords)
        u_lag, v_lag = interpolate_mac_velocity((u, v), weights, grid)
        assert jnp.allclose(u_lag, 1.0, atol=0.05)
        assert jnp.allclose(v_lag, 0.0, atol=0.05)

    def test_spread_conserves_momentum(self, grid, circle_coords):
        """Total spread forcing should approximately equal total Lagrangian forcing.

        The IBM spreading operator satisfies: sum(spread) ≈ sum(dU_lag).
        This follows from Peskin partition of unity and the dv factors
        cancelling between interpolation and spreading.
        """
        n_markers = 20
        dU = (jnp.ones(n_markers), jnp.zeros(n_markers))
        weights = compute_tensor_weights_mac(grid, circle_coords)
        spread = spread_mac_forcing(dU, weights, grid)
        total_eul = float(jnp.sum(spread[0]))
        total_lag = float(jnp.sum(dU[0]))
        assert abs(total_eul - total_lag) / abs(total_lag) < 0.05


class TestApplyIBMForcing:
    def test_reduces_velocity_at_body(self, grid, circle_coords):
        """IBM forcing should move velocity at body significantly toward target."""
        u = jnp.ones(grid.face_shape(0)) * 0.5
        v = jnp.zeros(grid.face_shape(1))
        U_tgt = (jnp.zeros(20), jnp.zeros(20))
        weights = compute_tensor_weights_mac(grid, circle_coords)
        # Velocity at body before forcing
        u_before, _ = interpolate_mac_velocity((u, v), weights, grid)
        vel_new, delta_U = apply_ibm_forcing((u, v), U_tgt, circle_coords, grid,
                                             weights=weights)
        # Velocity at body after forcing
        u_after, _ = interpolate_mac_velocity(vel_new, weights, grid)
        # After forcing, residual should be smaller than before
        assert float(jnp.max(jnp.abs(u_after))) < float(jnp.max(jnp.abs(u_before)))
        # delta_U should reflect the deficiency
        assert len(delta_U) == 2
        assert delta_U[0].shape == (20,)
