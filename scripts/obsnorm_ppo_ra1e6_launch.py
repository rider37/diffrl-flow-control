"""Discriminating experiment: PPO + running observation normalization
at Ra=1e6 on the production grid 256x192.  Identical to the PPO launcher
(scripts/rb_ppo_expand.py --ra 1e6: ENV_CONFIG_RA1E6 + PPO_V2_CONFIG, [64,64],
episode_length=400, ic_pool=range(15)) EXCEPT obs_norm=True.

Tests whether the chaotic-Ra PPO result persists under standard PPO
observation hygiene (the obs-normalization confound).  Reward is NOT
normalized (norm_obs only), to keep the return-variance variable clean.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiments.rb_control.config import ENV_CONFIG_RA1E6, PPO_V2_CONFIG
from experiments.rb_control.train_ppo import train as train_ppo

TRAIN_POOL = list(range(15))
SEEDS = [42, 123, 456, 517, 1223]


def run(seed):
    env_cfg = {**ENV_CONFIG_RA1E6}
    ppo_cfg = {
        **PPO_V2_CONFIG,
        'seed': seed,
        'episode_length': env_cfg.get('episode_actions', 400),
        'ic_pool_seeds': TRAIN_POOL,
        'obs_norm': True,
        'output_dir': f'experiments/rb_control/output_obsnorm_ppo_ra1e6_s{seed}',
    }
    print(f"\n{'='*70}\nPPO+obs-norm Ra=1e6 256x192 seed={seed}\n{'='*70}", flush=True)
    t0 = time.time()
    train_ppo(env_cfg, ppo_cfg)
    print(f"PPO+obs-norm Ra=1e6 s{seed} done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    # optional CLI seeds for resume, e.g.  python ...launch.py 123 456
    seeds = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else SEEDS
    overall = time.time()
    for seed in seeds:
        run(seed)
    print(f"\nALL obs-norm PPO Ra=1e6 {seeds} done in {(time.time()-overall)/60:.1f} min", flush=True)
