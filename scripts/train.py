"""Train one policy of the article.

    python scripts/train.py cylinder shac re100 --seed 42
    python scripts/train.py cylinder ppo re500 --seed 1223
    python scripts/train.py rayleigh_benard shac ra1e5 --seed 42
    python scripts/train.py rayleigh_benard ppo ra1e6 --seed 42
    python scripts/train.py rayleigh_benard ppo_obsnorm ra1e6 --seed 42     # Section 5.4
    python scripts/train.py rayleigh_benard shac_horizon40 ra1e4 --seed 42  # Section 5.4

The environment settings are read from configs/<problem>/env_<case>.json and the
algorithm settings from configs/<problem>/<algorithm>_<case>.json.  The run is
written to runs/<problem>/<algorithm>_<case>_s<seed>/ (history.json and
checkpoints/final.npz).  The article uses seeds 42 123 456 517 1223.
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)

from diffrl_flow_control.configs import load_config, run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('problem', choices=['cylinder', 'rayleigh_benard'])
    ap.add_argument('algorithm', help='shac, ppo, ppo_obsnorm or shac_horizon<h>')
    ap.add_argument('case', help='re100, re200, re400, re500 (cylinder); ra1e4, ra1e5, ra1e6 (rayleigh_benard)')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()

    env_cfg = load_config(a.problem, f'env_{a.case}')
    alg_cfg = load_config(a.problem, f'{a.algorithm}_{a.case}')
    alg_cfg['seed'] = a.seed
    alg_cfg['output_dir'] = run_dir(a.problem, f'{a.algorithm}_{a.case}', a.seed)

    if a.problem == 'cylinder':
        if a.algorithm == 'shac':
            from diffrl_flow_control.cylinder.train_shac import train
        else:
            from diffrl_flow_control.cylinder.ppo import train
    else:
        if a.algorithm.startswith('shac'):
            from diffrl_flow_control.rayleigh_benard.train_shac import train
        else:
            from diffrl_flow_control.rayleigh_benard.ppo import train
    print(f'{a.problem} {a.algorithm} {a.case} seed {a.seed} -> {alg_cfg["output_dir"]}', flush=True)
    train(env_cfg, alg_cfg)


if __name__ == '__main__':
    main()
