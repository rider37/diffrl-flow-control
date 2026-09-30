"""Held-out evaluation of the Ra=1e4 SHAC horizon sweep (h = 20, 40, 80, 160; seed 42, 50 epochs).

Evaluates the final (epoch-50) and best-training checkpoints of each horizon on the
held-out ICs 100-104 with the Table 4 protocol (last-half mean Nu against the
matched-window uncontrolled Nu of results/evaluation/rayleigh_benard_uncontrolled.json), and
records the training curve summary.  Writes results/checks/horizon_sweep.json
Run from the repository root.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import json
import numpy as np
from diffrl_flow_control.configs import load_config
ENV_RA1E4 = load_config('rayleigh_benard', 'env_ra1e4')
from diffrl_flow_control.rayleigh_benard.env import RBControlEnv
from diffrl_flow_control.rayleigh_benard.heldout import eval_on_ic, load_params_flexible

ICS = [100, 101, 102, 103, 104]
bl = np.array([r['last_half'] for r in json.load(open('results/evaluation/rayleigh_benard_uncontrolled.json'))['1e4']])
env = RBControlEnv(dict(ENV_RA1E4))
W, E = ENV_RA1E4['warmup_steps'], ENV_RA1E4['episode_actions']
out = {}
for h in (20, 40, 80, 160):
    d = f'runs/rayleigh_benard/shac_horizon{h}_ra1e4_s42'
    hist = json.load(open(d + '/history.json'))
    nu = np.array(hist['nu_mean'])
    rec = dict(train_nu_first10=float(nu[:10].mean()), train_nu_last10=float(nu[-10:].mean()), train_nu_min=float(nu.min()),
               train_reduction_last10_pct=float((1 - nu[-10:].mean() / hist['nu_baseline']) * 100))
    for tag, ck in (('final', 'final.npz'), ('best', 'best.npz')):
        params, _ = load_params_flexible(f'{d}/checkpoints/{ck}')
        nl = np.array([float(np.mean(eval_on_ic(env, params['actor'], ic, W, E, env.action_size)[E // 2:])) for ic in ICS])
        rec[f'heldout_{tag}_reduction_pct'] = float(((bl - nl) / bl * 100).mean())
        rec[f'heldout_{tag}_nu_late_per_ic'] = nl.tolist()
    out[h] = rec
    print(h, {k: round(v, 3) for k, v in rec.items() if not isinstance(v, list)}, flush=True)
json.dump(out, open('results/checks/horizon_sweep.json', 'w'), indent=1)
