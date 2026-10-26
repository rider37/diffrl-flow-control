"""Train one policy of the article.

    python scripts/train.py cylinder shac re100 --seed 42
    python scripts/train.py cylinder ppo re500 --seed 1223
    python scripts/train.py rayleigh_benard shac ra1e5 --seed 42
    python scripts/train.py rayleigh_benard ppo ra1e6 --seed 42
    python scripts/train.py rayleigh_benard ppo_obsnorm ra1e6 --seed 42     # Section 5.4
    python scripts/train.py rayleigh_benard shac_horizon40 ra1e4 --seed 42  # Section 5.4
    python scripts/train.py rayleigh_benard bptt ra1e5 --seed 42            # Section 4.4
    python scripts/train.py rayleigh_benard gippo ra1e5 --seed 42           # Section 4.4
    python scripts/train.py rayleigh_benard bptt ra1e5 --seed 42 --episode-actions 480 [--updates 120]   # Fig. 6
    python scripts/train.py rayleigh_benard shac ra1e5 --seed 42 --episode-actions 480                   # Fig. 6

The environment settings are read from configs/<problem>/env_<case>.json and the
algorithm settings from configs/<problem>/<algorithm>_<case>.json; full-episode
BPTT starts from the SHAC settings and GI-PPO from the PPO settings
(src/diffrl_flow_control/baselines/).  The run is written to
runs/<problem>/<name>_<case>_s<seed>/ (history.json and checkpoints/final.npz),
where <name> is the algorithm, extended by _ep<N> and _u<U> for the training
episode and update count of Fig. 6.  The article uses seeds 42 123 456 517 1223.

XLA GPU autotuning is switched off before JAX is imported, as in every run of
the article: with autotuning on, jitted backward passes were wrong by about a
factor of two in some processes on our GPUs.
"""
import argparse
import os
import sys

_flags = os.environ.get('XLA_FLAGS', '')
if 'xla_gpu_autotune_level' not in _flags:
    os.environ['XLA_FLAGS'] = (_flags + ' --xla_gpu_autotune_level=0').strip()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)

from diffrl_flow_control.configs import load_config, run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('problem', choices=['cylinder', 'rayleigh_benard'])
    ap.add_argument('algorithm', help='shac, ppo, ppo_obsnorm, shac_horizon<h>, bptt or gippo')
    ap.add_argument('case', help='re100, re200, re400, re500 (cylinder); ra1e4, ra1e5, ra1e6 (rayleigh_benard)')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--episode-actions', type=int, default=None,
                    help='shac, bptt: training episode in actions at the SHAC training interval (Fig. 6)')
    ap.add_argument('--updates', type=int, default=None,
                    help='bptt: fixed number of updates instead of SHAC\'s CFD budget (Fig. 6)')
    a = ap.parse_args()
    env_cfg = load_config(a.problem, f'env_{a.case}')

    if a.algorithm == 'gippo':
        from diffrl_flow_control.baselines import gippo
        out = run_dir(a.problem, f'gippo_{a.case}', a.seed)
        print(f'{a.problem} gippo {a.case} seed {a.seed} -> {out}', flush=True)
        gippo.train(a.problem, a.case, a.seed, out_dir=out)
        return

    name = a.algorithm
    if a.algorithm == 'bptt':
        from diffrl_flow_control.baselines import bptt
        alg_cfg = bptt.configure(env_cfg, load_config(a.problem, f'shac_{a.case}'), a.episode_actions, a.updates)
    else:
        alg_cfg = load_config(a.problem, f'{a.algorithm}_{a.case}')
        if a.episode_actions is not None:                 # SHAC with a longer training episode (Fig. 6)
            assert a.algorithm == 'shac' and a.episode_actions % alg_cfg['horizon'] == 0
            alg_cfg['reset_every'] = a.episode_actions // alg_cfg['horizon']
    if a.episode_actions is not None:
        name += f'_ep{a.episode_actions}'
    if a.updates is not None:
        name += f'_u{a.updates}'
    alg_cfg['seed'] = a.seed
    alg_cfg['output_dir'] = run_dir(a.problem, f'{name}_{a.case}', a.seed)

    if a.algorithm.startswith('shac') or a.algorithm == 'bptt':
        if a.problem == 'cylinder':
            from diffrl_flow_control.cylinder import train_shac as trainer
        else:
            from diffrl_flow_control.rayleigh_benard import train_shac as trainer
        if a.algorithm == 'bptt':
            trainer.shac_actor_loss = bptt.actor_loss(a.problem)
    elif a.problem == 'cylinder':
        from diffrl_flow_control.cylinder import ppo as trainer
    else:
        from diffrl_flow_control.rayleigh_benard import ppo as trainer
    print(f'{a.problem} {name} {a.case} seed {a.seed} -> {alg_cfg["output_dir"]}', flush=True)
    trainer.train(env_cfg, alg_cfg)


if __name__ == '__main__':
    main()
