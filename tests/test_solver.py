"""Grid, operators and time stepping of the solver."""
import jax
import jax.numpy as jnp
import numpy as np

from diffrl_flow_control.solver.grid import MACGrid
from diffrl_flow_control.solver.operators import divergence, gradient_p, diff_axis_padded_nu, vorticity_2d
from diffrl_flow_control.solver.schemes import solve_pressure_fft


def test_grid_shapes_and_coordinates():
    g = MACGrid((8, 4), (2.0, 1.0))
    assert g.ndim == 2 and g.spacings == (0.25, 0.25)
    assert g.face_shape(0) == (9, 4) and g.face_shape(1) == (8, 5)
    xc, yc = g.cell_centers()
    np.testing.assert_allclose(xc, 0.125 + 0.25 * np.arange(8))
    xu, _ = g.face_centers(0)
    np.testing.assert_allclose(xu, 0.25 * np.arange(9))


def test_divergence_of_uniform_flow_is_zero():
    g = MACGrid((16, 8), (2.0, 1.0))
    u = jnp.ones(g.face_shape(0)); v = jnp.zeros(g.face_shape(1))
    np.testing.assert_allclose(divergence((u, v), g), 0.0, atol=1e-6)


def test_pressure_gradient_of_linear_field():
    g = MACGrid((16, 8), (2.0, 1.0))
    xc, _ = g.cell_centers()
    p = jnp.broadcast_to(3.0 * xc[:, None], g.shape)
    gx, gy = gradient_p(p, g)
    np.testing.assert_allclose(gx[1:-1], 3.0, rtol=1e-5)
    np.testing.assert_allclose(gy, 0.0, atol=1e-5)


def test_second_derivative_of_quadratic():
    g = MACGrid((32, 4), (1.0, 1.0))
    xc, _ = g.cell_centers()
    f = jnp.broadcast_to((xc ** 2)[:, None], g.shape)
    fp = jnp.pad(f, ((1, 1), (0, 0)), mode='edge')
    d2 = diff_axis_padded_nu(fp, 0, g)
    np.testing.assert_allclose(d2[1:-1], 2.0, rtol=1e-3)


def test_neumann_poisson_solution():
    g = MACGrid((32, 16), (2.0, 1.0))
    xc, yc = g.cell_centers()
    X, Y = jnp.meshgrid(xc, yc, indexing='ij')
    p_exact = jnp.cos(jnp.pi * X / 2.0) * jnp.cos(jnp.pi * Y)
    lap = -(jnp.pi ** 2) * (0.25 + 1.0) * p_exact
    p = solve_pressure_fft(lap, g)
    np.testing.assert_allclose(p - p.mean(), p_exact - p_exact.mean(), atol=5e-3)


def test_vorticity_of_solid_rotation():
    g = MACGrid((16, 16), (1.0, 1.0))
    xu, yu = g.face_centers(0); xv, yv = g.face_centers(1)
    u = -jnp.broadcast_to(yu[None, :], g.face_shape(0))
    v = jnp.broadcast_to(xv[:, None], g.face_shape(1))
    w = vorticity_2d((u, v), g)
    np.testing.assert_allclose(w[1:-1, 1:-1], 2.0, rtol=1e-5)
