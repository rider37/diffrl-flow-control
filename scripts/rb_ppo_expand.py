"""RB PPO seed-expansion at Ra=1e5 / 1e6.

Verbatim generalisation of the canonical `phase3_ppo_ra1e6_expand.py`
(main repo) to either chaotic Ra: same PPO_V2_CONFIG, same multi-IC pool,
same episode_length = env episode_actions, same output-dir naming
(`output_phase3_ppo_ra1e{5,6}_s{seed}`).  Only the Ra selector is added so
the +2-seed dispatcher can address 1e5 and 1e6 with one module.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiments.rb_control.config import (
    ENV_CONFIG_RA1E5, ENV_CONFIG_RA1E6, PPO_V2_CONFIG,
)
from experiments.rb_control.train_ppo import train as train_ppo

TRAIN_POOL = list(range(15))
_ENVS = {'1e5': ENV_CONFIG_RA1E5, '1e6': ENV_CONFIG_RA1E6}


def run(ra, seed):
    assert ra in _ENVS, ra
    env_cfg = {**_ENVS[ra]}
    ppo_cfg = {
        **PPO_V2_CONFIG,
        'seed': seed,
        'episode_length': env_cfg.get('episode_actions', 400),
        'output_dir': f'experiments/rb_control/output_phase3_ppo_ra{ra}_s{seed}',
        'ic_pool_seeds': TRAIN_POOL,
    }
    print(f"\n{'='*70}\nPPO Ra={ra} seed={seed}\n{'='*70}", flush=True)
    t0 = time.time()
    train_ppo(env_cfg, ppo_cfg)
    print(f"PPO Ra={ra} s{seed} done in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ra', choices=['1e5', '1e6'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    run(a.ra, a.seed)
