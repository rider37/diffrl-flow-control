"""
Control configuration for 2D cylinder AFC benchmark.

t_end=200 gives ~40 vortex shedding cycles at Re=100 (St~0.167, period~6.0).
Sufficient for baseline Cd/Cl statistics and RL episode evaluation.
"""

CONTROL = {
    'time_control': {
        'max_cfl': 0.3,
        'adaptive_dt': True,
        't_end': 200.0,
        'save_every': 2000,
        'diag_every': 100,
    },

    'diagnostics': {
        'forces': True,
        'monitor': ['ke', 'div'],
        'probes': [
            [3.0, 2.0],    # 1D downstream, centerline
            [5.0, 2.0],    # 3D downstream
            [7.0, 2.0],    # 5D downstream
            [10.0, 2.0],   # 8D downstream
        ],
    },

    'output': {
        'dir': 'cases/cylinder_afc/output',
        'file': 'run.h5',
    },
}
