"""Held-out evaluation of the Ra=1e4 SHAC horizon sweep (h = 20, 40, 80, 160; seed 42, 50 epochs).

Evaluates the final (epoch-50) and best-training checkpoints of each horizon on the
held-out ICs 100-104 with the Table 4 protocol (last-half mean Nu against the
matched-window uncontrolled Nu of results/rb_heldout_matched_baselines.json), and
records the training curve summary.  Writes results/verification/v08_horizon_sweep_heldout.json
Run from the repository root.
"""
import os, sys, json
sys.path[:0] = ['.', 'src', 'scripts']
import numpy as np
from experiments.rb_control.config import ENV_CONFIG
from jax_cfd.envs.rb_control_env import RBControlEnv
from rb_eval_common import eval_on_ic, load_params_flexible

ICS = [100, 101, 102, 103, 104]
bl = np.array([r['last_half'] for r in json.load(open('results/rb_heldout_matched_baselines.json'))['1e4']])
env = RBControlEnv(dict(ENV_CONFIG))
W, E = ENV_CONFIG['warmup_steps'], ENV_CONFIG['episode_actions']
out = {}
for h in (20, 40, 80, 160):
    d = f'experiments/rb_control/output_optA1_h{h}_s42'
    hist = json.load(open(d + '/history.json'))
    nu = np.array(hist['nu_mean'])
    rec = dict(train_nu_first10=float(nu[:10].mean()), train_nu_last10=float(nu[-10:].mean()), train_nu_min=float(nu.min()),
               train_reduction_last10_pct=float((1 - nu[-10:].mean() / hist['nu_baseline']) * 100))
    for tag, ck in (('final', 'fans_final_epoch_0050.npz'), ('best', 'fans_best.npz')):
        params, _ = load_params_flexible(f'{d}/checkpoints/{ck}')
        nl = np.array([float(np.mean(eval_on_ic(env, params['actor'], ic, W, E, env.action_size)[E // 2:])) for ic in ICS])
        rec[f'heldout_{tag}_reduction_pct'] = float(((bl - nl) / bl * 100).mean())
        rec[f'heldout_{tag}_nu_late_per_ic'] = nl.tolist()
    out[h] = rec
    print(h, {k: round(v, 3) for k, v in rec.items() if not isinstance(v, list)}, flush=True)
json.dump(out, open('results/verification/v08_horizon_sweep_heldout.json', 'w'), indent=1)
