"""
Physics configuration for 2D cylinder AFC benchmark.

Default: Re=100 (laminar, Rabault et al. 2019).

Re variants from the literature:
  Re=100   Rabault et al. 2019, ~8% drag reduction (PPO baseline)
  Re=200   Tang et al. 2020,    ~17% drag reduction
  Re=400   Tang et al. 2020,    ~39% drag reduction
  Re=1000  Wang et al. 2024,    ~46.5% drag reduction (DF-DRL, JFM)
  Re=2000  Varela et al. 2022

To change Re, modify PHYSICS['Re'] or override at import time:
    from cases.cylinder_afc import CONFIG
    CONFIG['Re'] = 400
"""

PHYSICS = {
    'solver': 'incompressible_ns',
    'reference': {'U': 1.0, 'L': 1.0, 'rho': 1.0},
    'Re': 100,

    'turbulence': {'model': 'laminar'},

    'schemes': {
        'time': 'rk3',
        'advection': 'muscl',
        'pressure': 'fft',
    },

    'boundaries': {
        'x_min': {'type': 'inlet_parabolic', 'Um': 1.5},
        'x_max': {'type': 'outlet_zero_gradient'},
        'y_min': {'type': 'wall_no_slip'},
        'y_max': {'type': 'wall_no_slip'},
    },

    'initial_conditions': {
        'velocity': [1.0, 0.0],
        'pressure': 0.0,
    },
}
