"""
Domain configuration for 2D cylinder AFC benchmark.

Rabault et al. (2019, JFM) standard setup:
  - 22D x 4.1D rectangular channel
  - Cylinder D=1.0 at (2.0, 2.0), offset 0.05D below centerline (4.1/2 = 2.05)
  - Grid: 512 x 96 (~23 cells per D)

Reference:
  Rabault et al., "Artificial neural networks trained through deep reinforcement
  learning discover control strategies for active flow control", JFM 861, 2019.
"""

DOMAIN = {
    'domain': {
        'ndim': 2,
        'size': [22.0, 4.1],           # 22D x 4.1D channel
        'cells': [512, 96],            # ~23 cells per D
    },
    'bodies': [
        {
            'type': 'cylinder',
            'radius': 0.5,              # D = 1.0
            'center': [2.0, 2.0],       # 0.05D below centerline (2.05)
            'n_points': 128,
        },
    ],
}
