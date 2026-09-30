"""Eddy-viscosity closure of the momentum and scalar diffusion terms.

All flows of the article are laminar, so the closure returns zero; it is kept as
a function so that the diffusion terms are evaluated exactly as in the runs.
"""
from typing import Any, Dict

import jax.numpy as jnp

from diffrl_flow_control.solver.grid import MACGrid
from diffrl_flow_control.solver.state import FluidState


def laminar_viscosity(state: FluidState, grid: MACGrid, params: Dict[str, Any]) -> jnp.ndarray:
    """Zero eddy viscosity at the cell centres (laminar flow)."""
    return jnp.zeros(grid.cell_shape())
