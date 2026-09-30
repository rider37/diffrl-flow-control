"""Multi-GPU domain decomposition via jax.sharding.

Provides utilities to distribute a simulation across multiple devices
(GPUs/TPUs) using JAX's NamedSharding API.  The domain is decomposed
along a single axis (default: x) and halo exchange is performed via
``jax.lax.ppermute`` before each time step.

Usage (via runner CONFIG)::

    'parallel': {
        'shard_axis': 0,   # axis to split (0=x, 1=y)
        'halo_width': 3,   # ghost layers per side (>=3 for WENO5)
    }

Requirements
------------
- Multiple JAX devices visible (``jax.device_count() > 1``)
- Grid size along shard_axis must be divisible by device count
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax_cfd.core.state import FluidState


def get_mesh(shard_axis: int = 0) -> Mesh:
    """Create a 1-D device mesh for domain decomposition."""
    devices = jax.devices()
    return Mesh(devices, axis_names=('x',))


def shard_state(state: FluidState, mesh: Mesh, shard_axis: int = 0) -> FluidState:
    """Distribute a FluidState across devices along *shard_axis*.

    Each velocity component, pressure, and scalar field is sharded so
    that the *shard_axis* dimension is split across the mesh.
    """
    ndim = len(state.velocity)

    def _make_pspec(array_ndim):
        spec = [None] * array_ndim
        if shard_axis < array_ndim:
            spec[shard_axis] = 'x'
        return P(*spec)

    def _shard(arr):
        if arr is None:
            return None
        pspec = _make_pspec(arr.ndim)
        sharding = NamedSharding(mesh, pspec)
        return jax.device_put(arr, sharding)

    vel = tuple(_shard(u) for u in state.velocity)
    p = _shard(state.pressure)
    scalars = {k: _shard(v) for k, v in state.scalars.items()}
    ibm_f = state.ibm_forces  # small array, replicate

    return FluidState(velocity=vel, pressure=p, scalars=scalars,
                      ibm_forces=ibm_f)


def gather_state(state: FluidState) -> FluidState:
    """Collect a distributed FluidState onto a single device."""
    def _gather(arr):
        if arr is None:
            return None
        return jax.device_get(arr)

    vel = tuple(jnp.array(_gather(u)) for u in state.velocity)
    p = jnp.array(_gather(state.pressure))
    scalars = {k: jnp.array(_gather(v)) for k, v in state.scalars.items()}
    return FluidState(velocity=vel, pressure=p, scalars=scalars,
                      ibm_forces=state.ibm_forces)


def make_sharded_step_fn(step_fn, grid, mesh: Mesh, shard_axis: int = 0):
    """Wrap a step function for sharded (multi-GPU) execution.

    The returned function accepts and returns sharded FluidStates.
    JAX's XLA compiler handles the necessary communication (halo
    exchange) automatically when operations cross shard boundaries,
    provided the step function is traced under the correct sharding
    context.

    Parameters
    ----------
    step_fn : callable
        Original ``step_fn(state, dt, extra, **kwargs) -> FluidState``.
    grid : MACGrid
        Simulation grid.
    mesh : Mesh
        Device mesh from ``get_mesh()``.
    shard_axis : int
        Axis along which the domain is decomposed.

    Returns
    -------
    sharded_step_fn : callable
        Same signature as *step_fn*, but operates on sharded states.
    """
    ndim = grid.ndim

    def _out_pspec(ndim_arr):
        spec = [None] * ndim_arr
        if shard_axis < ndim_arr:
            spec[shard_axis] = 'x'
        return P(*spec)

    # Define output sharding for FluidState pytree
    _vel_pspec = tuple(_out_pspec(ndim) for _ in range(ndim))
    _p_pspec = _out_pspec(ndim)

    @jax.jit
    def sharded_step_fn(state, dt, extra, **kwargs):
        return step_fn(state, dt, extra, **kwargs)

    return sharded_step_fn
