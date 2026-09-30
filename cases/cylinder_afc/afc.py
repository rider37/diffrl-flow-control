"""
Active Flow Control configuration for 2D cylinder benchmark.

Matches Rabault et al. (2019, JFM) standard setup exactly:

Actuators:
  2 ZNMF synthetic jets at theta = 90/270 deg.
  ZNMF: Q_top + Q_bottom = 0 (single scalar action).
  EMA action smoothing alpha=0.1 per CFD step.

Sensors:
  151 pressure probes in 3 groups (Rabault 2019):
    - Wake grid:     9x7 = 63 probes
    - Upstream grid:  4x4 = 16 probes
    - Cylinder rings: 2x36 = 72 probes

Reward:
  r = -(Cd - Cd_baseline) - 0.2 * |mean(Cl)|

Timing:
  50 CFD steps per RL action ~ 0.25 convective time units.
  80 actions per episode ~ 20 convective time units ~ 6 shedding cycles.
"""

import numpy as np

AFC = {
    # ── Actuators: ZNMF synthetic jets ───────────────────────────────
    'jets': {
        'n_jets': 2,
        'thetas': [np.pi / 2, 3 * np.pi / 2],   # 90 deg, 270 deg
        'width': 0.15,                            # Gaussian width of jet
        'Q_max': 0.5,                             # max volume flow rate
    },

    # ── Sensors: pressure probes (Rabault 2019) ────────────────────
    'probes': {
        'layout': 'rabault2019',   # 3-group placement (151 probes total)
        'obs_type': 'pressure',    # pressure at cell centers
    },

    # ── RL timing ────────────────────────────────────────────────────
    'timing': {
        'dt': 0.005,
        'action_interval': 50,     # CFD steps per RL action (0.25 time units)
        'episode_length': 80,      # actions per episode (20 time units)
        'warmup_steps': 30000,     # develop vortex shedding (150 time units)
        'baseline_steps': 10000,   # measure uncontrolled Cd (50 time units)
        'ema_alpha': 0.1,          # EMA action smoothing (Rabault 2019)
    },

    # ── Reward function ──────────────────────────────────────────────
    'reward': {
        'type': 'rabault2019',     # r = -(Cd - Cd0) - cl_weight*|mean(Cl)|
        'cl_weight': 0.2,
    },
}
