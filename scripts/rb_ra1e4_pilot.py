"""Ra=1e4 multi-IC PPO training (seeds 42 123 456 517 1223 by default).

Train pool: 15 random ICs (seeds 0..14)
Test pool:  5 random ICs (seeds 100..104), disjoint

Post-training evaluation: scripts/eval_rb_heldout.py.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiments.rb_control.config import ENV_CONFIG, PPO_V2_CONFIG
from experiments.rb_control.train_ppo import train as train_ppo


TRAIN_POOL = list(range(15))   # seeds 0..14
TEST_POOL = list(range(100, 105))  # seeds 100..104


def run_ppo(seed):
    env_cfg = {**ENV_CONFIG}  # uses defaults: warmup=3000, baseline=1000
    ppo_cfg = {
        **PPO_V2_CONFIG,
        'seed': seed,
        'output_dir': f'experiments/rb_control/output_phase2_ppo_ra1e4_s{seed}',
        'ic_pool_seeds': TRAIN_POOL,
    }
    print(f"\n{'='*70}\nPhase 2 PPO Ra=1e4 seed={seed}\n{'='*70}")
    t0 = time.time()
    train_ppo(env_cfg, ppo_cfg)
    print(f"PPO s{seed} done in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    overall_t = time.time()

    # PPO only: the published Ra=1e4 SHAC is scripts/train_rb1e4_shac1000.py
    for seed in [int(a) for a in sys.argv[1:]] or [42, 123, 456, 517, 1223]:
        run_ppo(seed)

    print(f"\n{'='*70}\nPhase 2 ALL RUNS COMPLETE — {(time.time()-overall_t)/60:.1f} min")
    print("Next: scripts/eval_rb_heldout.py evaluates the final checkpoints on the held-out pool")
