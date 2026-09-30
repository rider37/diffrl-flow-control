"""Re=500 PPO retrain, plateau grid 1280x240, calibrated interface eps=0.75,
with VELOCITY observation (151 near-wake velocity probes, 302-dim).

Matches retrain_re500_shac_velocity.py so the Re=500 SHAC-vs-PPO comparison
uses the same near-wake velocity sensing as Re=100/200/400.

Output dir: output_eps075vel_ppo_re500_1280x240_s{seed}
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cases.cylinder_afc import make_env_config
from experiments.afc_cylinder.config import PPO_CONFIG
from experiments.afc_cylinder.train_ppo import train as train_ppo

GRID = (1280, 240)
RE = 500
SEEDS = [42, 123, 456, 517, 1223]


def make_env():
    env_cfg = make_env_config(Re=RE, cells=GRID, ibm_eps_factor=0.75)
    env_cfg['jet_method'] = 'surface_bc'
    env_cfg['soft_reset_prob'] = 0.8
    env_cfg['obs_type'] = 'velocity'        # <-- unify with Re<=400
    return env_cfg


def run_ppo(seed, n_updates=100):
    env_cfg = make_env()
    ppo_cfg = {
        **PPO_CONFIG,
        'seed': seed,
        'n_updates': n_updates,
        'output_dir': (f'experiments/afc_cylinder/'
                       f'output_eps075vel_ppo_re{RE}_1280x240_s{seed}'),
    }
    print(f"\n{'='*70}\nPPO VELOCITY Re={RE} GRID={GRID[0]}x{GRID[1]} "
          f"eps=0.75 seed={seed} n_updates={n_updates}\n{'='*70}", flush=True)
    t0 = time.time()
    train_ppo(env_cfg, ppo_cfg)
    print(f"PPO Re={RE} vel s{seed} done in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=None,
                    help='single seed; default runs all 5')
    ap.add_argument('--n_updates', type=int, default=100)
    a = ap.parse_args()
    seeds = [a.seed] if a.seed is not None else SEEDS
    overall = time.time()
    for s in seeds:
        run_ppo(s, a.n_updates)
    print(f"\nALL PPO velocity Re={RE} done in "
          f"{(time.time()-overall)/60:.1f} min", flush=True)
