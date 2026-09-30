"""Shared fixtures for JAX-CFD unit tests."""
import sys
from pathlib import Path

import pytest
import jax.numpy as jnp

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jax_cfd.core.mac_grid import MACGrid
from jax_cfd.core.state import FluidState


@pytest.fixture
def grid_2d():
    """A small 2D uniform MAC grid."""
    return MACGrid((16, 16), (1.0, 1.0))


@pytest.fixture
def grid_3d():
    """A small 3D uniform MAC grid."""
    return MACGrid((8, 8, 8), (1.0, 1.0, 1.0))


@pytest.fixture
def state_2d(grid_2d):
    """A simple 2D uniform-flow state."""
    g = grid_2d
    u = jnp.ones(g.face_shape(0))
    v = jnp.zeros(g.face_shape(1))
    p = jnp.zeros(g.cell_shape())
    return FluidState(velocity=(u, v), pressure=p, scalars={}, ibm_forces=None)


@pytest.fixture
def state_3d(grid_3d):
    """A simple 3D uniform-flow state."""
    g = grid_3d
    u = jnp.ones(g.face_shape(0))
    v = jnp.zeros(g.face_shape(1))
    w = jnp.zeros(g.face_shape(2))
    p = jnp.zeros(g.cell_shape())
    return FluidState(velocity=(u, v, w), pressure=p, scalars={}, ibm_forces=None)
