"""Held-out evaluation of the final Rayleigh-Benard policies (Table 4).

Every policy runs deterministically from each held-out IC (seeds 100-104)
after the environment warm-up, for one evaluation episode of `episode_actions`
actions; the stored quantity is the mean Nu over the last half of the episode.
Reductions are computed against the matched uncontrolled references by
scripts/analysis/heldout_statistics.py.

Final checkpoints (seeds 42 123 456 517 1223):
  runs/rayleigh_benard/{shac,ppo}_ra{Ra}_s{seed}/checkpoints/final.npz

Usage: python scripts/evaluate/rayleigh_benard_heldout.py [--ra 1e4 1e5 1e6] [--algo ppo shac] [--seeds ...]
                                                          [--checkpoint epoch_0010]
Writes results/evaluation/rayleigh_benard_heldout.json: {Ra: {algo: {seed: {ckpt, nu_late_per_ic}}}};
a checkpoint other than final.npz is stored under the key algo@<checkpoint>.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json, time, argparse
import numpy as np
from diffrl_flow_control.configs import load_config
ENV_RA1E4 = load_config('rayleigh_benard', 'env_ra1e4')
ENV_RA1E5 = load_config('rayleigh_benard', 'env_ra1e5')
ENV_RA1E6 = load_config('rayleigh_benard', 'env_ra1e6')
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
from diffrl_flow_control.rayleigh_benard.heldout import load_params_flexible, eval_on_ic, TEST_POOL, SEEDS

RA_CFG = {'1e4': ENV_RA1E4, '1e5': ENV_RA1E5, '1e6': ENV_RA1E6}
OUT = 'results/evaluation/rayleigh_benard_heldout.json'


def ckpt(ra, algo, seed, name='final'):
    return f'runs/rayleigh_benard/{algo}_ra{ra}_s{seed}/checkpoints/{name}.npz'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ra', nargs='+', default=list(RA_CFG))
    ap.add_argument('--algo', nargs='+', default=['ppo', 'shac'])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--checkpoint', default='final', help='checkpoint name, e.g. epoch_0010 (default: final)')
    a = ap.parse_args()
    key = (lambda al: al) if a.checkpoint == 'final' else (lambda al: f'{al}@{a.checkpoint}')
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for ra in a.ra:
        cfg = RA_CFG[ra]; env = RBControlEnv({**cfg})
        warmup, ep_actions, adim = cfg['warmup_steps'], cfg['episode_actions'], env.action_size
        print(f"\n=== Ra={ra}  warmup={warmup}  episode={ep_actions} actions ===", flush=True)
        for algo in a.algo:
            for s in a.seeds:
                ck = ckpt(ra, algo, s, a.checkpoint); t0 = time.time()
                params, _ = load_params_flexible(ck)
                nl = [float(np.mean(eval_on_ic(env, params['actor'], ic, warmup, ep_actions, adim)[ep_actions // 2:]))
                      for ic in TEST_POOL]
                out.setdefault(ra, {}).setdefault(key(algo), {})[str(s)] = {'ckpt': ck, 'nu_late_per_ic': nl}
                print(f"  {key(algo)} s{s}: Nu_late per IC {np.round(nl, 3)}  [{(time.time()-t0)/60:.1f} min]", flush=True)
                os.makedirs('results', exist_ok=True)
                json.dump(out, open(OUT, 'w'), indent=1)
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
