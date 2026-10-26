"""Training-interval check of the Ra=1e6 SHAC policies (Section 5.4).

Held-out protocol of scripts/evaluate/rayleigh_benard_heldout.py (ICs 100-104, final
checkpoint, last-half mean Nu, reduction against the matched-window uncontrolled Nu of
results/evaluation/rayleigh_benard_uncontrolled.json, mean over ICs), evaluated at the
environment interval (300 steps, 400 actions; reproduces Table 4) and at the SHAC
training interval (150 steps, 800 actions; same physical duration).
Writes results/checks/training_interval_ra1e6.json.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json, time
import numpy as np
from diffrl_flow_control.configs import load_config
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
from diffrl_flow_control.rayleigh_benard.heldout import load_params_flexible, eval_on_ic, TEST_POOL

SEEDS = [42, 123, 456, 517, 1223]
ENV = load_config('rayleigh_benard', 'env_ra1e6')
bl = np.array([r['last_half'] for r in json.load(open('results/evaluation/rayleigh_benard_uncontrolled.json'))['1e6']])
out = {}
for ai, ep in ((300, 400), (150, 800)):
    cfg = {**ENV, 'action_interval': ai, 'episode_actions': ep}
    env = RBControlEnv(cfg)
    per_seed, nl_all = {}, {}
    for s in SEEDS:
        params, _ = load_params_flexible(f'runs/rayleigh_benard/shac_ra1e6_s{s}/checkpoints/final.npz')
        t0 = time.time()
        nl = np.array([float(np.mean(eval_on_ic(env, params['actor'], ic, cfg['warmup_steps'], ep, env.action_size)[ep // 2:]))
                       for ic in TEST_POOL])
        red = (bl - nl) / bl * 100.0
        per_seed[s] = float(red.mean()); nl_all[s] = nl.tolist()
        print(f'[interval {ai}] s{s}: {red.mean():+.2f}%  [{(time.time() - t0) / 60:.1f} min]', flush=True)
    r = np.array(list(per_seed.values()))
    out[ai] = {'per_seed': per_seed, 'nu_late_per_ic': nl_all, 'mean': float(r.mean()), 'sd_sample': float(r.std(ddof=1))}
    print(f'[interval {ai}] mean {r.mean():+.2f} +- {r.std(ddof=1):.2f}', flush=True)
d = np.array([out[150]['per_seed'][s] - out[300]['per_seed'][s] for s in SEEDS])
out['diff_150_minus_300'] = {'mean': float(d.mean()), 'per_seed': d.tolist()}
os.makedirs('results/checks', exist_ok=True)
json.dump(out, open('results/checks/training_interval_ra1e6.json', 'w'), indent=1)
print(f'SUMMARY diff (150-300) = {d.mean():+.2f} pp, per seed {np.round(d, 2)}')
