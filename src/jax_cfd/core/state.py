import jax.numpy as jnp
from typing import NamedTuple, Dict, Optional


class FluidState(NamedTuple):
    """
    N-dimensional master state for the CFD solver.
    Registered as a JAX PyTree via NamedTuple so jax.tree_map works transparently.

    Fields
    ------
    velocity    : tuple of face-centred arrays — (u, v[, w])
    pressure    : cell-centred pressure array
    scalars     : dict of cell-centred scalar fields, e.g. {'T': temp_array}
    ibm_forces  : integrated body forces (ndim,) for IBM cases; None otherwise
    """
    velocity:   tuple
    pressure:   jnp.ndarray
    scalars:    Dict[str, jnp.ndarray] = {}
    ibm_forces: Optional[jnp.ndarray] = None
