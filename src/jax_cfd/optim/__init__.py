"""
Optimisation toolkit for JAX-CFD.

Modules
-------
parameterize : Brinkman alpha, density filter, Sobolev gradient
simulate     : Differentiable simulation + IFT steady-state solve
optimizers   : MMA, projected GD, Adam, multi-resolution cascade
continuation : Penalisation continuation schedules
mcmo         : Multi-Condition Multi-Objective (Chebyshev, Pareto, MLP)
"""

from jax_cfd.optim.parameterize import (
    brinkman_alpha,
    make_kappa_field,
    density_filter,
    smooth_max,
    sobolev_filter,
)
from jax_cfd.optim.simulate import (
    differentiable_simulate,
    differentiable_simulate_segmented,
    steady_state_solve,
)
from jax_cfd.optim.optimizers import (
    topology_optimize,
    mma_update,
    adam_init,
    adam_update,
    lbfgs_init,
    lbfgs_step,
    multi_resolution_optimize,
)
from jax_cfd.optim.continuation import (
    make_continuation_schedule,
    ContinuationState,
)
from jax_cfd.optim.mcmo import (
    chebyshev_scalarization,
    chebyshev_augmented,
    sample_weight_vectors,
    sample_weight_random,
    update_utopia,
    extract_pareto_front,
    hypervolume_2d,
    init_mlp_params,
    mlp_forward,
    mcmo_optimize_direct,
)

__all__ = [
    'brinkman_alpha', 'make_kappa_field', 'density_filter', 'smooth_max',
    'sobolev_filter',
    'differentiable_simulate', 'differentiable_simulate_segmented',
    'steady_state_solve',
    'topology_optimize', 'mma_update', 'adam_init', 'adam_update',
    'lbfgs_init', 'lbfgs_step', 'multi_resolution_optimize',
    'make_continuation_schedule', 'ContinuationState',
    'chebyshev_scalarization', 'chebyshev_augmented',
    'sample_weight_vectors', 'sample_weight_random',
    'update_utopia', 'extract_pareto_front', 'hypervolume_2d',
    'init_mlp_params', 'mlp_forward', 'mcmo_optimize_direct',
]
