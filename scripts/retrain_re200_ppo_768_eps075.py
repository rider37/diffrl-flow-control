"""Re=200 PPO retrain on plateau grid 768x144 (seeds 42 123 456 517 1223; pass a subset on the command line)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cases.cylinder_afc import make_env_config
from experiments.afc_cylinder.config import PPO_CONFIG
from experiments.afc_cylinder.train_ppo import train as train_ppo

GRID = (768, 144); RE = 200


def make_env():
    env_cfg = make_env_config(Re=RE, cells=GRID, ibm_eps_factor=0.75)
    env_cfg['jet_method'] = 'surface_bc'
    env_cfg['soft_reset_prob'] = 0.8
    env_cfg['obs_type'] = 'velocity'
    return env_cfg


def run_ppo(seed, n_updates=100):
    env_cfg = make_env()
    ppo_cfg = {
        **PPO_CONFIG,
        'seed': seed,
        'n_updates': n_updates,
        'output_dir': (f'experiments/afc_cylinder/'
                       f'output_eps075_ppo_re{RE}_768x144_s{seed}'),
    }
    print(f"\n{'='*70}\nCanonical PPO Re={RE} GRID={GRID[0]}x{GRID[1]} "
          f"seed={seed} n_updates={n_updates}\n{'='*70}", flush=True)
    t0 = time.time()
    train_ppo(env_cfg, ppo_cfg)
    print(f"PPO Re={RE} s{seed} done in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    overall = time.time()
    for seed in [int(a) for a in sys.argv[1:]] or [42, 123, 456, 517, 1223]:
        run_ppo(seed)
    print(f"\nALL Re={RE} 768x144 PPO done in "
          f"{(time.time()-overall)/60:.1f} min")
