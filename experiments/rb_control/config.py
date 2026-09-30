"""Configuration for 2D Rayleigh-Benard AFC training (PPO + FANS).

Reference: Markmann et al. 2025 (arXiv:2504.12000)
"""

import numpy as np


# ── Environment config (Markmann 2025) ────────────────────────────────

ENV_CONFIG = {
    'Ra': 1e4,
    'Pr': 0.7,
    'Lx': 2 * np.pi,
    'Ly': 2.0,
    'Nx': 96,
    'Ny': 64,

    'T_hot': 1.0,
    'T_cold': 0.0,
    'perturb_amp': 0.05,

    'n_segments': 12,
    'amplitude': 0.75,

    # dt=0.025 causes CFL divergence on 96x64 (dy=0.031, U_max>1.25).
    # dt=0.01 → CFL≈0.2, stable.  Same physical time per action (1.5 t.u.).
    'dt': 0.01,
    'action_interval': 150,
    'episode_actions': 200,
    'warmup_steps': 3000,
    'baseline_steps': 1000,

    'n_probes_x': 48,
    'n_probes_y': 8,
}


# ── PPO config (Markmann 2025 matching) ───────────────────────────────

PPO_CONFIG = {
    'actor_hidden_dims': [64, 64],
    'critic_hidden_dims': [64, 64],
    'activation': 'relu',

    'lr': 1e-3,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_eps': 0.2,
    'n_epochs': 10,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,

    'episodes_per_update': 4,
    'episode_length': 100,
    'n_updates': 50,
    'seed': 42,

    'log_every': 1,
    'save_every': 10,
    'snapshot_every': 10,
    'output_dir': 'experiments/rb_control/output_ppo',
}


# ── FANS config ───────────────────────────────────────────────────────

FANS_CONFIG = {
    'horizon': 20,
    'action_interval': 150,
    'segment_length': 15,

    'gamma': 0.99,
    'action_penalty': 0.0,

    'actor_lr': 3e-4,
    'critic_lr': 1e-3,
    'max_grad_norm': 5.0,

    'hidden_dims': [64, 64],

    'n_epochs': 100,
    'critic_update_epochs': 5,
    'reset_every': 25,
    'seed': 42,

    'log_every': 5,
    'save_every': 50,
    'snapshot_every': 25,
    'output_dir': 'experiments/rb_control/output_fans',
}


# ── Quick test configs ────────────────────────────────────────────────

TEST_ENV_CONFIG = {
    **ENV_CONFIG,
    'Nx': 32,
    'Ny': 16,
    'warmup_steps': 50,
    'baseline_steps': 30,
    'action_interval': 10,
    'episode_actions': 10,
    'n_probes_x': 8,
    'n_probes_y': 4,
}

TEST_PPO_CONFIG = {
    **PPO_CONFIG,
    'n_updates': 3,
    'episodes_per_update': 2,
    'episode_length': 10,
    'actor_hidden_dims': [32, 32],
    'critic_hidden_dims': [32, 32],
    'n_epochs': 3,
    'output_dir': 'experiments/rb_control/output_ppo_test',
}

TEST_FANS_CONFIG = {
    **FANS_CONFIG,
    'horizon': 5,
    'action_interval': 10,
    'segment_length': 5,
    'hidden_dims': [32, 32],
    'n_epochs': 3,
    'critic_update_epochs': 2,
    'output_dir': 'experiments/rb_control/output_fans_test',
}


# ── V2 configs (improved training) ──────────────────────────────────

# PPO v2: closer to Markmann spec (8 eps × 200 steps = 1600 trans/update)
PPO_V2_CONFIG = {
    **PPO_CONFIG,
    'episodes_per_update': 8,
    'episode_length': 200,
    'n_updates': 100,
    'output_dir': 'experiments/rb_control/output_ppo_v2',
}

# FANS v2: no periodic reset (continuous state advance), more epochs
FANS_V2_CONFIG = {
    **FANS_CONFIG,
    'reset_every': 0,
    'n_epochs': 200,
    'snapshot_every': 50,
    'output_dir': 'experiments/rb_control/output_fans_v2',
}


# ── PD controller config ────────────────────────────────────────────

PD_CONFIG = {
    'Kp': 1.0,
    'Kd': 0.1,
    'Nu_target_ratio': 0.5,     # target Nu = ratio × Nu_baseline
    'n_steps': 200,             # number of control actions
    'grid_search': True,        # run Kp×Kd grid search
    'Kp_range': [0.1, 0.3, 1.0, 3.0, 10.0],
    'Kd_range': [0.01, 0.03, 0.1, 0.3, 1.0],
    'seed': 42,
    'output_dir': 'experiments/rb_control/output_pd',
}


# ── Ra=10⁵ configs ──────────────────────────────────────────────────

ENV_CONFIG_RA1E5 = {
    'Ra': 1e5,
    'Pr': 0.7,
    'Lx': 2 * np.pi,
    'Ly': 2.0,
    'Nx': 128,
    'Ny': 96,
    'T_hot': 1.0,
    'T_cold': 0.0,
    'perturb_amp': 0.05,
    'n_segments': 12,
    'amplitude': 0.75,
    'dt': 0.004,
    'action_interval': 150,
    'episode_actions': 250,
    'warmup_steps': 6000,
    'baseline_steps': 2000,
    'n_probes_x': 64,
    'n_probes_y': 12,
}

PPO_CONFIG_RA1E5 = {
    **PPO_CONFIG,
    'episodes_per_update': 8,
    'episode_length': 250,
    'n_updates': 100,
    'actor_hidden_dims': [96, 96],
    'critic_hidden_dims': [96, 96],
    'output_dir': 'experiments/rb_control/output_ppo_ra1e5',
}

FANS_CONFIG_RA1E5 = {
    **FANS_CONFIG,
    'horizon': 30,
    'segment_length': 20,
    'actor_lr': 3e-4,
    'action_penalty': 0.0,
    'n_epochs': 200,
    'hidden_dims': [96, 96],
    'reset_every': 0,
    'output_dir': 'experiments/rb_control/output_fans_ra1e5',
}

PD_CONFIG_RA1E5 = {
    **PD_CONFIG,
    'output_dir': 'experiments/rb_control/output_pd_ra1e5',
}


# ── Ra=10⁶ configs ──────────────────────────────────────────────────

ENV_CONFIG_RA1E6 = {
    'Ra': 1e6,
    'Pr': 0.7,
    'Lx': 2 * np.pi,
    'Ly': 2.0,
    'Nx': 256,
    'Ny': 192,
    'T_hot': 1.0,
    'T_cold': 0.0,
    'perturb_amp': 0.05,
    'n_segments': 12,
    'amplitude': 0.75,
    'dt': 0.0015,
    'action_interval': 300,
    'episode_actions': 400,
    'warmup_steps': 15000,
    'baseline_steps': 5000,
    'n_probes_x': 128,
    'n_probes_y': 24,
}

PPO_CONFIG_RA1E6 = {
    **PPO_CONFIG,
    'episodes_per_update': 8,
    'episode_length': 400,
    'n_updates': 100,
    'actor_hidden_dims': [128, 128],
    'critic_hidden_dims': [128, 128],
    'output_dir': 'experiments/rb_control/output_ppo_ra1e6',
}

FANS_CONFIG_RA1E6 = {
    **FANS_CONFIG,
    'horizon': 50,
    'segment_length': 30,
    'actor_lr': 1e-4,
    'action_penalty': 0.0,
    'n_epochs': 100,
    'hidden_dims': [128, 128],
    'max_grad_norm': 3.0,
    'reset_every': 0,
    'output_dir': 'experiments/rb_control/output_fans_ra1e6',
}

PD_CONFIG_RA1E6 = {
    **PD_CONFIG,
    'output_dir': 'experiments/rb_control/output_pd_ra1e6',
}
