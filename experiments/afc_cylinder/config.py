"""Configuration for 2D cylinder AFC training (PPO + FANS).

Environment config is imported from cases/cylinder_afc/ (Rabault et al. 2019
standard benchmark). Algorithm-specific hyperparameters are defined here.
"""

from cases.cylinder_afc import make_env_config

# ── Environment config (Rabault 2019 standard) ─────────────────────────

ENV_CONFIG = make_env_config(Re=100, cells=(1024, 192))
ENV_CONFIG['jet_method'] = 'surface_bc'  # IBM surface BC jet injection
ENV_CONFIG['soft_reset_prob'] = 0.8      # 80% soft / 20% hard reset


# ── PPO config (Rabault 2019 matching) ─────────────────────────────────

PPO_CONFIG = {
    # Network — separate actor/critic (Rabault 2019)
    'actor_hidden_dims': [512, 512],   # 2×512 (Rabault)
    'critic_hidden_dims': [32, 32],    # 2×32 (Rabault, separate small critic)
    'activation': 'relu',              # ReLU (Rabault/Tensorforce default)

    # PPO hyperparameters
    'lr': 1e-3,              # Rabault (was 3e-4)
    'gamma': 0.99,
    'gae_lambda': 0.97,      # Rabault (was 0.95)
    'clip_eps': 0.2,
    'n_epochs': 25,           # Rabault (was 10)
    'subsample_ratio': 0.2,   # 20% per epoch (Rabault)
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,

    # Training — multi-episode batching (Rabault 2019)
    'episodes_per_update': 20,    # 20 episodes per PPO update
    'episode_length': 80,         # 80 actions per episode
    'n_updates': 100,             # 100 updates × 20 ep = 2000 episodes total
    'seed': 42,

    # Logging
    'log_every': 1,
    'save_every': 10,
    'snapshot_every': 10,     # flow field snapshot every N updates
    'output_dir': 'experiments/afc_cylinder/output_ppo_rabault',
}


# ── Quick test config (smaller grid, fewer steps) ───────────────────────

TEST_ENV_CONFIG = make_env_config(Re=100, cells=(88, 24))
TEST_ENV_CONFIG.update({
    'warmup_steps': 500,
    'baseline_steps': 200,
    'action_interval': 20,
    'episode_actions': 20,
    'n_ibm_pts': 64,
    'jet_method': 'surface_bc',
    'soft_reset_prob': 0.0,
})

TEST_PPO_CONFIG = {
    **PPO_CONFIG,
    'n_updates': 3,
    'episodes_per_update': 2,
    'episode_length': 20,
    'actor_hidden_dims': [64, 64],
    'critic_hidden_dims': [16, 16],
    'n_epochs': 5,
    'output_dir': 'experiments/afc_cylinder/output_ppo_test',
}


# ── FANS config (1st-order gradient RL through DiffFlow) ───────────────

FANS_CONFIG = {
    # SHAC truncation
    'horizon': 48,             # h: cover ~2 shedding cycles (48×50×0.005=12.0 ≈ 2×T_shed)
    'action_interval': 50,     # CFD steps per RL action
    'segment_length': 10,      # checkpoint segment for memory efficiency

    # Discount
    'gamma': 0.99,

    # Reward
    'cl_weight': 0.1,          # Cl penalty to suppress asymmetric flow
    'action_penalty': 0.0,     # no action penalty (let gradient decide)

    # Optimizer
    'actor_lr': 3e-4,          # more aggressive (exact gradient → stable)
    'critic_lr': 1e-3,
    'max_grad_norm': 5.0,      # less clipping (allow larger updates early)

    # Network
    'hidden_dims': [256, 256],

    # Training
    'n_epochs': 300,
    'critic_update_epochs': 5,  # critic SGD steps per actor update
    'reset_every': 50,         # reset state to baseline every N epochs (anti-drift)
    'seed': 42,

    # Logging & data saving
    'log_every': 5,
    'save_every': 50,
    'snapshot_every': 25,      # flow field snapshot + image every N epochs
    'output_dir': 'experiments/afc_cylinder/output_fans_sbc_h48',
}

TEST_FANS_CONFIG = {
    **FANS_CONFIG,
    'horizon': 4,
    'action_interval': 20,
    'segment_length': 5,
    'hidden_dims': [64, 64],
    'n_epochs': 5,
    'critic_update_epochs': 2,
    'output_dir': 'experiments/afc_cylinder/output_fans_test',
}


# ── Re=200 configs ───────────────────────────────────────────────────

ENV_CONFIG_RE200 = make_env_config(Re=200, cells=(512, 96))
ENV_CONFIG_RE200['jet_method'] = 'surface_bc'
ENV_CONFIG_RE200['soft_reset_prob'] = 0.8

PPO_CONFIG_RE200 = {
    **PPO_CONFIG,
    'output_dir': 'experiments/afc_cylinder/output_ppo_re200',
}

FANS_CONFIG_RE200 = {
    **FANS_CONFIG,
    'output_dir': 'experiments/afc_cylinder/output_fans_re200',
}


# ── Re=400 configs (matched to canonical SHAC Re=400 setup) ─────────

ENV_CONFIG_RE400 = make_env_config(Re=400, cells=(768, 144))
ENV_CONFIG_RE400['jet_method'] = 'surface_bc'
ENV_CONFIG_RE400['soft_reset_prob'] = 0.8

PPO_CONFIG_RE400 = {
    **PPO_CONFIG,
    'output_dir': 'experiments/afc_cylinder/output_ppo_re400',
}

FANS_CONFIG_RE400 = {
    **FANS_CONFIG,
    'output_dir': 'experiments/afc_cylinder/output_fans_re400',
}


# ── Re=500 configs ───────────────────────────────────────────────────

ENV_CONFIG_RE500 = make_env_config(Re=500, cells=(768, 144))
ENV_CONFIG_RE500['jet_method'] = 'surface_bc'
ENV_CONFIG_RE500['soft_reset_prob'] = 0.8

PPO_CONFIG_RE500 = {
    **PPO_CONFIG,
    'output_dir': 'experiments/afc_cylinder/output_ppo_re500',
}

FANS_CONFIG_RE500 = {
    **FANS_CONFIG,
    'actor_lr': 1e-4,           # lower lr for higher Re (gradient noise)
    'output_dir': 'experiments/afc_cylinder/output_fans_re500',
}
