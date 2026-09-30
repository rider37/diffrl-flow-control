"""Held-out evaluation of the final Rayleigh-Benard policies (Table 4).

Every policy runs deterministically from each held-out IC (seeds 100-104)
after the environment warm-up, for one evaluation episode of `episode_actions`
actions; the stored quantity is the mean Nu over the last half of the episode.
Reductions are computed against the matched uncontrolled references by
scripts/heldout_matched_stats.py.

Final checkpoints (seeds 42 123 456 517 1223):
  PPO  Ra=1e4           output_phase2_ppo_ra1e4_s{seed}/checkpoints/ckpt_0100.npz
  PPO  Ra=1e5, 1e6      output_phase3_ppo_ra{Ra}_s{seed}/checkpoints/ckpt_0100.npz
  SHAC Ra=1e4           output_phase3_shac_ra1e4_s{seed}/checkpoints/fans_final_epoch_1000.npz
  SHAC Ra=1e5, 1e6      output_wm_shac_ra{Ra}_s{seed}/checkpoints/fans_final_epoch_1000.npz

Usage: python scripts/eval_rb_heldout.py [--ra 1e4 1e5 1e6] [--algo ppo shac] [--seeds ...]
Writes results/rb_heldout.json: {Ra: {algo: {seed: {ckpt, nu_late_per_ic}}}}
"""
import os, sys, json, time, argparse
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
from experiments.rb_control.config import ENV_CONFIG, ENV_CONFIG_RA1E5, ENV_CONFIG_RA1E6
from jax_cfd.envs.rb_control_env import RBControlEnv
from rb_eval_common import load_params_flexible, eval_on_ic, TEST_POOL, SEEDS

RA_CFG = {'1e4': ENV_CONFIG, '1e5': ENV_CONFIG_RA1E5, '1e6': ENV_CONFIG_RA1E6}
OUT = 'results/rb_heldout.json'


def ckpt(ra, algo, seed):
    d = 'experiments/rb_control/'
    if algo == 'ppo':
        ph = 'phase2' if ra == '1e4' else 'phase3'
        return d + f'output_{ph}_ppo_ra{ra}_s{seed}/checkpoints/ckpt_0100.npz'
    run = 'phase3_shac' if ra == '1e4' else 'wm_shac'
    return d + f'output_{run}_ra{ra}_s{seed}/checkpoints/fans_final_epoch_1000.npz'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ra', nargs='+', default=list(RA_CFG))
    ap.add_argument('--algo', nargs='+', default=['ppo', 'shac'])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    a = ap.parse_args()
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for ra in a.ra:
        cfg = RA_CFG[ra]; env = RBControlEnv({**cfg})
        warmup, ep_actions, adim = cfg['warmup_steps'], cfg['episode_actions'], env.action_size
        print(f"\n=== Ra={ra}  warmup={warmup}  episode={ep_actions} actions ===", flush=True)
        for algo in a.algo:
            for s in a.seeds:
                ck = ckpt(ra, algo, s); t0 = time.time()
                params, _ = load_params_flexible(ck)
                nl = [float(np.mean(eval_on_ic(env, params['actor'], ic, warmup, ep_actions, adim)[ep_actions // 2:]))
                      for ic in TEST_POOL]
                out.setdefault(ra, {}).setdefault(algo, {})[str(s)] = {'ckpt': ck, 'nu_late_per_ic': nl}
                print(f"  {algo} s{s}: Nu_late per IC {np.round(nl, 3)}  [{(time.time()-t0)/60:.1f} min]", flush=True)
                os.makedirs('results', exist_ok=True)
                json.dump(out, open(OUT, 'w'), indent=1)
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
